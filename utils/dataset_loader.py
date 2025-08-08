"""
IntelliForm — Dataset Loader (XFUND/FUNSD style)
=================================================

WHAT THIS MODULE DOES
---------------------
Provides PyTorch `Dataset` and `DataLoader` utilities to feed training and
validation data into the IntelliForm models.

It:
1) Loads preprocessed JSON annotation files from `data/{split}/annotations/*.json`.
   Each JSON should contain tokens, bboxes, labels (ints), and optional page ids.
2) Applies the LayoutLMv3 tokenizer (and optional image features if used).
3) Normalizes/validates bbox coordinates to the expected LayoutLMv3 range.
4) (Optional) Builds graph connectivity per sample via `utils.graph_builder`
   to support the GNN layer in the classifier.
5) Yields batched tensors ready for the model’s forward pass.

EXPECTED ANNOTATION FORMAT
--------------------------
A single document JSON contains at minimum:
{
  "id": "doc_0001",
  "tokens": ["Full", "Name", ":" , "John", "Doe"],
  "bboxes": [[x0,y0,x1,y1], ...],      # normalized or raw; normalization handled here
  "labels": [1,1,0, 0,0],              # per-token label ids, with O=0 or ignore_index
  "page_ids": [1,1,1, 1,1]             # optional (for multi-page cases)
}

INPUTS
------
- split: str ("train" | "val")
- root_dir: str (default: "data")
- label_map_path: str (default: "data/labels.json")
- tokenizer_name_or_path: str (e.g., "microsoft/layoutlmv3-base")
- use_gnn: bool (default: True)
- graph_cfg: dict (k, radius, strategy, etc.)
- max_seq_length: int (for tokenization/truncation)
- image_usage: bool (set True if using pixel values pipeline)

OUTPUT (PER BATCH)
------------------
Dict with standard LayoutLMv3 keys:
  input_ids: LongTensor [B, T]
  attention_mask: LongTensor [B, T]
  bbox: LongTensor [B, T, 4]
  labels: LongTensor [B, T] (training only; may be absent at inference)
Optional:
  pixel_values: FloatTensor [B, C, H, W] (if image branch used)
  graph: dict with edge_index/edge_attr/num_nodes for GNN

KEY CLASSES / FUNCTIONS
-----------------------
- class FormDataset(Dataset):
    Loads samples from `data/{split}/annotations`, applies tokenization and packing.

- def make_dataloaders(...):
    Returns (train_loader, val_loader) with appropriate `collate_fn` to:
      * pad sequences,
      * stack bbox/labels,
      * attach graph dicts (if use_gnn=True).

DEPENDENCIES
------------
- torch, torch.utils.data
- transformers (LayoutLMv3 tokenizer)
- numpy/pandas (light preprocessing)
- utils.graph_builder (optional, for edge construction)

INTERACTIONS
------------
- Used by: `scripts/train_classifier.py` (training/val), `utils/llmv3_infer.py` (batched inference)
- Consumes: `data/train/annotations/*.json`, `data/val/annotations/*.json`, `data/labels.json`

NOTES & TODOs
-------------
- Ensure bbox normalization (0..1000 for LayoutLMv families) is consistent with tokenizer config.
- Add caching for graph edges to speed repeated epochs.
- Add on-the-fly data augmentation if needed (e.g., small bbox jitter).

"""
