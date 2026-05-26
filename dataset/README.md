# Dataset Preprocessing

## Directory Structure

```
dataset/
├── preprocessing/
│   ├── process_amazon/
│   │   ├── 1split_dataset.ipynb
│   │   ├── 2get_metadata.ipynb
│   │   ├── 3text_feature_extractor.ipynb
│   │   └── 4image_feature_extractor.ipynb
│   └── process_microlens/
│       ├── 1split_dataset.ipynb
│       ├── 2text_feature_extractor.ipynb
│       └── 3image_feature_extractor.ipynb
├── amazon/
│   └── raw/
│       ├── meta_Beauty.json.gz
│       ├── ratings_Beauty.csv
└── microlens/
    └── raw/
        ├── MicroLens-100k_pairs.csv
        ├── MicroLens-100k_title_en.csv
        └── tags_to_summary.csv
```

---

## Amazon Beauty Dataset

### Raw Files

Download the following files and place them in `dataset/Beauty/raw/`:

| File | Description |
|------|-------------|
| `meta_Beauty.json.gz` | Product metadata (title, description, category, etc.) |
| `ratings_Beauty.csv` | User rating data |

### Download Link

- [Amazon Review Data (2014)](http://jmcauley.ucsd.edu/data/amazon/)

### Preprocessing Pipeline

Run the notebooks in `preprocessing/process_amazon/` in order:

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `1split_dataset.ipynb` | Split data into train/validation/test sets |
| 2 | `2get_metadata.ipynb` | Extract metadata from `meta_Beauty.json.gz` |
| 3 | `3text_feature_extractor.ipynb` | Extract text features |
| 4 | `4image_feature_extractor.ipynb` | Extract image features |

---

## MicroLens Dataset

### Raw Files

Download the following files and place them in `dataset/microlens/raw/`:

| File | Description |
|------|-------------|
| `MicroLens-100k_pairs.csv` | User-video interaction pairs |
| `MicroLens-100k_title_en.csv` | Video titles (English) |
| `tags_to_summary.csv` | Tags to summary mapping |

### Download Link

- [MicroLens Dataset](https://github.com/westlake-repl/MicroLens)

### Preprocessing Pipeline

Run the notebooks in `preprocessing/process_microlens/` in order:

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `1split_dataset.ipynb` | Split data into train/validation/test sets |
| 2 | `2text_feature_extractor.ipynb` | Extract text features |
| 3 | `3image_feature_extractor.ipynb` | Extract image features |

---

## Processed Data

After the preprocessing pipeline, the dataset is transformed into the following files for downstream training and evaluation:

- `dataset_name.train.inter`, `dataset_name.valid.inter`, `dataset_name.test.inter`: sequential interaction data for training, validation, and testing, respectively.
- `image.npy`: pre-extracted visual features of items.
- `text.npy`: pre-extracted textual features of items.

The `.inter` files contain user-item interaction sequences, while the `.npy` files provide pre-trained multi-modal item representations.

---