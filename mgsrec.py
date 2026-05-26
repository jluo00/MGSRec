import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from recbole.model.layers import TransformerEncoder

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

class MGSRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.temperature = config['temperature']

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.svd_q = config['svd_q']
        self.svd_transform_U = nn.Linear(self.svd_q, self.svd_q, bias=False)
        self.svd_transform_V = nn.Linear(self.svd_q, self.svd_q, bias=False)
        self.singular_scale = nn.Parameter(torch.ones(self.svd_q))

        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gnn_layers = config['gnn_layers']
        self.global_graph_construction(dataset, k=2, semantic_topk=config['semantic_topk'])
        self.get_svd()    

        self.lamada = config['lamada']

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.nce_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

        self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
        dim = self.plm_embedding.embedding_dim
        self.pvm_embedding = copy.deepcopy(dataset.pvm_embedding)
        dim_v = self.pvm_embedding.embedding_dim

        self.text_trs = MoEAdaptorLayer(config["n_exps"], [dim, self.hidden_size, self.hidden_size], 0.2)
        self.image_trs = MoEAdaptorLayer(config["n_exps"], [dim_v, self.hidden_size, self.hidden_size], 0.2)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def normalize_adj(self, adj: sp.spmatrix):
        adj = adj + sp.eye(adj.shape[0])
        degree = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(degree, -0.5, where=(degree != 0))
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

    def build_semantic_sim_matrix(self, embeddings: np.ndarray, top_k=5, threshold=0.9):
        N = embeddings.shape[0]
        if top_k == 0:
            return sp.coo_matrix((N, N)).tocsr()

        sim_matrix = cosine_similarity(embeddings)
        sim_matrix[sim_matrix <= threshold] = 0
        np.fill_diagonal(sim_matrix, 0)
        
        top_k_indices = np.argpartition(sim_matrix, -top_k, axis=1)[:, -top_k:]
        row_indices = np.arange(N)[:, np.newaxis]
        top_k_values = sim_matrix[row_indices, top_k_indices]
        
        non_zero_mask = top_k_values > 0
        
        broadcasted_rows = np.broadcast_to(row_indices, top_k_indices.shape)

        final_rows = broadcasted_rows[non_zero_mask]
        final_cols = top_k_indices[non_zero_mask]
        final_data = top_k_values[non_zero_mask]

        return sp.coo_matrix((final_data, (final_rows, final_cols)), shape=(N, N)).tocsr()

    def build_semantic_augmented_graph(self, rule_adj: sp.csr_matrix, semantic_sim: sp.csr_matrix, 
                                       beta: float, top_k: int):
        print("Starting semantic propagation...")
        propagated_adj = semantic_sim.T @ rule_adj + rule_adj @ semantic_sim
        
        propagated_adj.setdiag(0)
        propagated_adj.eliminate_zeros()
        
        print(f"Propagation created {propagated_adj.nnz} potential new edges.")

        # Pruning
        rows, cols, data = [], [], []
        # propagated_adj = propagated_adj - propagated_adj.multiply(rule_adj > 0) # only new edges, is not good.
        propagated_adj_lil = propagated_adj.tolil()

        for i in range(propagated_adj_lil.shape[0]):
            if len(propagated_adj_lil.rows[i]) > 0:
                row_data = propagated_adj_lil.data[i]
                row_indices = propagated_adj_lil.rows[i]
                if len(row_data) > top_k: # select topk
                    top_indices = np.argpartition(row_data, -top_k)[-top_k:]
                    for idx in top_indices:
                        rows.append(i)
                        cols.append(row_indices[idx])
                        data.append(row_data[idx])
                else:
                    rows.extend([i] * len(row_data))
                    cols.extend(row_indices)
                    data.extend(row_data)
        
        pruned_propagated_adj = sp.coo_matrix((data, (rows, cols)), shape=propagated_adj.shape)
        print(f"After pruning to top-{top_k}, {pruned_propagated_adj.nnz} new edges remain.")

        final_adj = rule_adj.tocoo() + beta * pruned_propagated_adj
        
        return final_adj.tocoo()


    def global_graph_construction(self, dataset, k=2, semantic_topk=5):
        grouped_data = defaultdict(list)

        for i in range(len(dataset.inter_feat)):
            uid = dataset.inter_feat['user_id'][i].item()
            history_len = dataset.inter_feat['item_length'][i].item()
            history_seq = dataset.inter_feat['item_id_list'][i][:history_len].tolist()
            target_item = dataset.inter_feat['item_id'][i].item()
            grouped_data[uid].append({
                'len': history_len,
                'seq': history_seq + [target_item]
            })

        user_seq_dict = {}
        for uid, sequences_info in grouped_data.items():
            longest_sequence_info = max(sequences_info, key=lambda x: x['len'])
            user_seq_dict[uid] = longest_sequence_info['seq']

        row, col, data = [], [], []
        for item_list in user_seq_dict.values():
            item_list_len = len(item_list)
            for item_idx in range(item_list_len - 1):
                target_num = min(k, item_list_len - item_idx - 1)
                if target_num > 0:
                    row.extend([item_list[item_idx]] * target_num)
                    col.extend(item_list[item_idx + 1: item_idx + 1 + target_num])
                    data.append(1 / np.arange(1, 1 + target_num))

        if not data:
            n_items = dataset.item_num - 1
            rule_adj = sp.csc_matrix((n_items, n_items))
        else:
            data = np.concatenate(data)
            row, col = np.array(row) - 1, np.array(col) - 1
            n_items = dataset.item_num - 1
            rule_adj = sp.csc_matrix((data, (row, col)), shape=(n_items, n_items))
        
        rule_adj = (rule_adj + rule_adj.T).tocsr()
        print(f"Rule_adj have {rule_adj.nnz} edges.")

        text_embed = dataset.plm_embedding.weight.numpy()[1:]
        image_embed = dataset.pvm_embedding.weight.numpy()[1:]
        text_sim_matrix = self.build_semantic_sim_matrix(text_embed, top_k=semantic_topk)
        image_sim_matrix = self.build_semantic_sim_matrix(image_embed, top_k=semantic_topk)
        semantic_sim_matrix = self.alpha * text_sim_matrix + (1 - self.alpha) * image_sim_matrix
        
        fused_adj_raw = self.build_semantic_augmented_graph(
            rule_adj, semantic_sim_matrix, beta=self.beta, top_k=5
        )
        
        fused_adj_norm = self.normalize_adj(fused_adj_raw).tocoo() 

        indices = torch.from_numpy(np.vstack((fused_adj_norm.row, fused_adj_norm.col))).long()
        values = torch.from_numpy(fused_adj_norm.data).float()
        
        self.norm_adj = torch.sparse_coo_tensor(
            indices,
            values,
            (n_items, n_items),
            dtype=torch.float32
        )

    def get_svd(self):
        svd_q = self.svd_q
        sparse_tensor = self.norm_adj.coalesce()
        print(f"Performing SVD on sparse tensor (keeping top {svd_q} components)...")
        
        with torch.no_grad():
            svd_u, s, svd_v = torch.svd_lowrank(sparse_tensor, q=svd_q)
            self.register_buffer('u_mul_s', svd_u @ torch.diag(s))
            self.register_buffer('vt', svd_v.T)
            
        print(f"SVD done. Matrix approximation rank: {svd_q}")
        print(f"Top 5 singular values: {s[:5].tolist()}")
        print(f"Singular value decay ratio: {s[-1]/s[0]:.4f}")

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = item_emb[item_seq] + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_emb = self.dropout(item_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(item_emb, extended_attention_mask, output_all_encoded_layers=True)

        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def info_nce(self, z_i, z_j, temp, sim='dot'):
        """
        Compute the InfoNCE loss
        """
        batch_size = z_i.size(0)
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            # Apply L2 normalization first, then compute dot product
            z = F.normalize(z, dim=1)  # Normalize each row
            sim_matrix = torch.mm(z, z.T) / temp
        elif sim == 'dot':
            sim_matrix = torch.mm(z, z.T) / temp
        else:
            raise ValueError("Similarity metric not supported.")

        # Positive sample pairs (z_i, z_j)
        sim_i_j = torch.diag(sim_matrix, batch_size)
        sim_j_i = torch.diag(sim_matrix, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # Negative sample pairs (all other samples)
        mask = torch.ones((N, N), dtype=bool, device=self.device).fill_diagonal_(0)
        mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = 0
        mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = 0
        negative_samples = sim_matrix[mask].reshape(N, -1)

        labels = torch.zeros(N, dtype=torch.long, device=self.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.nce_fct(logits, labels)
        return loss

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        text_embeddings = self.text_trs(self.plm_embedding.weight)
        image_embeddings = self.image_trs(self.pvm_embedding.weight)
        item_emb = text_embeddings + image_embeddings 

        # graph encoder
        graph_emb = self.gnn_emb(self.norm_adj.to(item_seq.device), item_emb)

        # trm encoder
        seq_output = self.forward(item_seq, graph_emb, item_seq_len)

        # ########## CL Loss #############
        unique_items = torch.unique(pos_items)
        graph_emb_noise = self.gnn_emb(self.norm_adj.to(item_seq.device), item_emb, noise=True)
        gcl_loss = self.info_nce(graph_emb[unique_items], graph_emb_noise[unique_items], 0.2, sim='cos')

        # ########## Rec Loss ############
        seq_fused_emb = F.normalize(seq_output, dim=1)
        item_emb = F.normalize(item_emb, dim=1) 

        logits = torch.matmul(seq_fused_emb, item_emb.transpose(0, 1)) / self.temperature
        loss = self.loss_fct(logits, pos_items)
        # print(f"loss: {loss.item():.6f}, gcl_loss: {gcl_loss.item():.6f}")

        return loss# + self.lamada * gcl_loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        text_embeddings = self.text_trs(self.plm_embedding.weight)
        image_embeddings = self.image_trs(self.pvm_embedding.weight)
        item_emb = text_embeddings + image_embeddings 

        # graph encoder
        graph_emb = self.gnn_emb(self.norm_adj.to(item_seq.device), item_emb)

        # trm encoder
        seq_output = self.forward(item_seq, graph_emb, item_seq_len)

        seq_fused_emb = F.normalize(seq_output, dim=1)
        item_emb = F.normalize(item_emb, dim=1) 

        scores = torch.matmul(seq_fused_emb, item_emb.transpose(0, 1))  # [B n_items]
        return scores  

    def gnn_emb(self, g, emb, noise=False):
        graph_emb = emb
        graph_emb_list = [graph_emb]
        
        for idx in range(self.gnn_layers):
            padding_row = graph_emb[0:1]
            data_to_process = graph_emb[1:]
            propagated_data = torch.sparse.mm(g, data_to_process)
            
            if noise:
                low_rank_features = self.vt @ data_to_process
                transformed_features = self.svd_transform_V(low_rank_features.T)
                
                scaled_features = transformed_features.T * self.singular_scale.view(-1, 1)
                scaled_features = self.svd_transform_U(scaled_features.T).T
                
                perturbation = self.u_mul_s @ scaled_features

                new_data = propagated_data + self.config['graph_noise'] * perturbation
            else:
                new_data = propagated_data
            
            graph_emb = torch.cat([padding_row, new_data], dim=0)
            graph_emb_list.append(graph_emb)
        
        graph_emb = torch.stack(graph_emb_list, dim=1).mean(1)
        return graph_emb

class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([MLPLayers([layers[0], layers[1], layers[2]], dropout, 'gelu') for _ in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs

        return multiple_outputs.sum(dim=-2)
  
class MLPLayers(nn.Module):
    """MLPLayers
    """
    def __init__(self, layers, dropout=0.0, activation="relu", 
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            activation_func = self.get_layer_act(self.activation)
            if activation_func is not None:
                mlp_modules.append(activation_func)
        self.mlp_layers = nn.Sequential(*mlp_modules)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def get_layer_act(self, act):
        ACT2E = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return ACT2E[act]

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)