import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset


class MGSRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight, self.plm_size)

        self.pvm_size = config['pvm_size']
        self.pvm_suffix = config['pvm_suffix']
        pvm_embedding_weight = self.load_pvm_embedding()
        self.pvm_embedding = self.weight2emb(pvm_embedding_weight, self.pvm_size)

    def load_plm_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.plm_suffix}')
        loaded_feat = np.load(feat_path, allow_pickle=True)

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[int(token)]
        return mapped_feat
    
    def load_pvm_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.pvm_suffix}')
        loaded_feat = np.load(feat_path, allow_pickle=True)

        mapped_feat = np.zeros((self.item_num, self.pvm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[int(token)]
        return mapped_feat

    def weight2emb(self, weight, emb_size):
        embedding = nn.Embedding(self.item_num, emb_size, padding_idx=0)
        embedding.weight.requires_grad = False
        embedding.weight.data.copy_(torch.from_numpy(weight))
        return embedding
