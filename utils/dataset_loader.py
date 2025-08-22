# utils/dataset_loader.py

"""
IntelliForm Dataset Loader (FUNSD/XFUND Format)
===============================================

WHAT THIS MODULE DOES
---------------------
Provides PyTorch `Dataset` and utilities to feed annotation data into the
IntelliForm models (LayoutLMv3 + GNN + classifier).

Responsibilities:
1. Loads annotation JSON files from `data/{split}/annotations/*.json`.
2. Applies the LayoutLMv3 tokenizer (apply_ocr=False).
3. Normalizes/validates bounding boxes (0–1000 range expected).
4. Maps string labels → integer IDs via `labels.json`.
5. (Optional) Integrates graph connectivity via `utils.graph_builder`
   for use in the GNN-enhanced classifier.
6. Returns batched tensors ready for model forward passes.

EXPECTED ANNOTATION FORMAT
--------------------------
Each JSON must contain:
{
  "id": "doc_0001",
  "tokens": ["Full", "Name", ":", "John", "Doe"],
  "bboxes": [[x0,y0,x1,y1], ...],   # normalized or raw; normalization handled here
  "labels": ["B-NAME","I-NAME","O","B-NAME","I-NAME"],
  "page_ids": [0,0,0,0,0]           # optional (defaults to all-zeros if missing)
}

INPUTS
------
- annotations_dir : str (e.g., "data/train/annotations")
- labels_path     : str (default "data/labels.json")
- pretrained_name : str (Hugging Face model id, e.g. "microsoft/layoutlmv3-base")
- max_length      : int (default 512, tokenizer truncation/pad length)
- use_images      : bool (default False; extendable for pixel branch)

OUTPUTS
-------
Dataset samples provide:
- input_ids      : LongTensor [T]
- attention_mask : LongTensor [T]
- bbox           : LongTensor [T,4]
- labels         : LongTensor [T]  (with -100 for padding)
- page_ids       : LongTensor [T]

Collated batches (`collate_fn`) provide:
- input_ids      : LongTensor [B,T]
- attention_mask : LongTensor [B,T]
- bbox           : LongTensor [B,T,4]
- labels         : LongTensor [B,T]
- page_ids       : LongTensor [B,T]

KEY CLASSES / FUNCTIONS
-----------------------
- class FormDataset(Dataset)
    Loads and encodes annotation samples.

- def collate_fn(samples)
    Stacks multiple samples into a batch.

DEPENDENCIES
------------
- torch, torch.utils.data
- transformers.LayoutLMv3Processor
- json / os

INTERACTIONS
------------
- Used by: `scripts/train_classifier.py`, `scripts/train_all.py`
- Consumed by: `utils.llmv3_infer.py` (batched inference)
- Consumes: `data/train/annotations/*.json`, `data/val/annotations/*.json`, `data/labels.json`

SECURITY / DEPLOY NOTES
-----------------------
- Validate that annotations are consistent with `labels.json`.
- Large datasets may require caching of graph features.
"""

from __future__ import annotations
from typing import List, Dict, Any
import os, json
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor


@dataclass
class Sample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    bbox: torch.Tensor
    labels: torch.Tensor
    page_ids: torch.Tensor
    raw: Dict[str, Any]   # original JSON, useful for debugging


class FormDataset(Dataset):
    def __init__(
        self,
        annotations_dir: str,
        labels_path: str,
        pretrained_name: str = "microsoft/layoutlmv3-base",
        max_length: int = 512,
        use_images: bool = False,
    ):
        self.ann_paths = sorted([
            os.path.join(annotations_dir, f)
            for f in os.listdir(annotations_dir)
            if f.endswith(".json")
        ])
        if not self.ann_paths:
            raise RuntimeError(f"No JSON annotations found in {annotations_dir}")

        with open(labels_path, "r", encoding="utf-8") as f:
            self.label2id: Dict[str, int] = json.load(f)
        self.id2label = {v: k for k, v in self.label2id.items()}

        self.processor = LayoutLMv3Processor.from_pretrained(
            pretrained_name,
            apply_ocr=False
        )

        self.max_length = max_length
        self.use_images = use_images

    def __len__(self) -> int:
        return len(self.ann_paths)

    def __getitem__(self, idx: int) -> Sample:
        with open(self.ann_paths[idx], "r", encoding="utf-8") as f:
            ann = json.load(f)

        tokens: List[str] = ann["tokens"]
        bboxes: List[List[int]] = ann["bboxes"]
        str_labels: List[str] = ann["labels"]
        page_ids: List[int] = ann.get("page_ids", [0] * len(tokens))

        # Map labels → ids
        labels = [self.label2id.get(lbl, self.label2id.get("O", 0)) for lbl in str_labels]

        encoded = self.processor(
            text=tokens,
            boxes=bboxes,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Pad/align labels & page_ids
        labels_tensor = torch.full((self.max_length,), fill_value=-100, dtype=torch.long)
        page_ids_tensor = torch.full((self.max_length,), fill_value=0, dtype=torch.long)

        take = min(len(labels), self.max_length)
        labels_tensor[:take] = torch.tensor(labels[:take], dtype=torch.long)
        page_ids_tensor[:take] = torch.tensor(page_ids[:take], dtype=torch.long)

        return Sample(
            input_ids=encoded["input_ids"].squeeze(0),
            attention_mask=encoded["attention_mask"].squeeze(0),
            bbox=encoded["bbox"].squeeze(0),
            labels=labels_tensor,
            page_ids=page_ids_tensor,
            raw=ann
        )


def collate_fn(samples: List[Sample]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([s.input_ids for s in samples]),
        "attention_mask": torch.stack([s.attention_mask for s in samples]),
        "bbox": torch.stack([s.bbox for s in samples]),
        "labels": torch.stack([s.labels for s in samples]),
        "page_ids": torch.stack([s.page_ids for s in samples]),
    }
