# utils/dataset_loader.py
"""
IntelliForm Dataset Loader (FUNSD/XFUND/Custom JSON)
====================================================

WHAT THIS MODULE DOES
---------------------
Provides PyTorch `Dataset` and utilities to feed annotation data into the
IntelliForm models (LayoutLMv3 + (optional) GNN + classifier).

Responsibilities:
1. Loads annotation JSON files from `data/{split}/annotations/*.json` OR HuggingFace
   `save_to_disk` folders (e.g., .../hf_train, .../hf_test).
2. Uses the LayoutLMv3 *tokenizer* path when `use_images=False` (text-only; fast).
3. Uses the LayoutLMv3 *processor* path when `use_images=True` (expects a PIL image).
4. Normalizes bounding boxes to the 0–1000 grid (auto-scales 0..1 → 0..1000, clamps).
5. Maps string labels → integer IDs via `labels.json` (guarantees 'O' exists).
6. Returns batched tensors ready for model forward passes.

EXPECTED ANNOTATION FORMAT (JSON)
---------------------------------
Each JSON must contain:
{
  "id": "doc_0001",
  "tokens": ["Full", "Name", ":", "John", "Doe"],
  "bboxes": [[x0,y0,x1,y1], ...],   # may be 0..1 or 0..1000; normalized here
  "labels": ["B-NAME","I-NAME","O","B-NAME","I-NAME"],
  "page_ids": [0,0,0,0,0]           # optional (defaults to zeros)
  # If use_images=True, also include: "image_path": "path/to/page.png"
}

INPUTS
------
- annotations_dir : str (e.g., "data/train/annotations" or "data/train/funsd/hf_train")
- labels_path     : str (default "data/labels.json")
- pretrained_name : str (e.g., "microsoft/layoutlmv3-base")
- max_length      : int (default 512)
- use_images      : bool (default False; enable only if images are provided)

OUTPUTS
-------
Sample (per item):
- input_ids      : LongTensor [T]
- attention_mask : LongTensor [T]
- bbox           : LongTensor [T,4]
- labels         : LongTensor [T]  (with -100 for non-first-subwords/special/pad)
- page_ids       : LongTensor [T]

Batch (via collate_fn):
- input_ids      : LongTensor [B,T]
- attention_mask : LongTensor [B,T]
- bbox           : LongTensor [B,T,4]
- labels         : LongTensor [B,T]
- page_ids       : LongTensor [B,T]

KEY CLASSES / FUNCTIONS
-----------------------
- class FormDataset(Dataset)  → Loads + encodes
- def collate_fn(samples)     → Batches tensors

DEPENDENCIES
------------
- torch, torch.utils.data
- transformers.AutoTokenizer (text-only)
- transformers.LayoutLMv3Processor (image+text)
- datasets (optional; HF save_to_disk)

INTERACTIONS
------------
- Used by: scripts/train_classifier.py, scripts/evaluate_test.py
- Consumed by: utils/llmv3_infer.py
- Consumes: JSON under data/*/annotations/*.json or HF folders under data/*/hf_*, plus data/labels.json

NOTES
-----
- We DO NOT pass `is_split_into_words` (your HF version rejects it). We align
  labels and page_ids using `encoded.word_ids()` instead.
"""

from __future__ import annotations
from typing import List, Dict, Any, Iterable, Optional
import os, json

from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, LayoutLMv3Processor

# Optional HF import (only used if we detect an HF folder)
try:
    from datasets import load_from_disk
    _HAS_HF = True
except Exception:
    _HAS_HF = False


# ----------------------------- Helpers -----------------------------

def _is_hf_folder(p: str) -> bool:
    """Heuristic: a Hugging Face 'save_to_disk' folder contains dataset_info.json."""
    return os.path.isdir(p) and os.path.exists(os.path.join(p, "dataset_info.json"))


def _iter_json_files(annotations_dir: str) -> Iterable[Dict[str, Any]]:
    """Yield annotation dicts from *.json files in a directory."""
    json_paths = sorted(
        os.path.join(annotations_dir, f)
        for f in os.listdir(annotations_dir)
        if f.endswith(".json")
    )
    if not json_paths:
        raise RuntimeError(f"No JSON annotations found in {annotations_dir}")

    for p in json_paths:
        with open(p, "r", encoding="utf-8") as f:
            yield json.load(f)


def _iter_hf_examples(hf_dir: str) -> Iterable[Dict[str, Any]]:
    """Yield examples from an HF 'save_to_disk' dataset normalized to our JSON schema."""
    if not _HAS_HF:
        raise RuntimeError("`datasets` is not installed. Run: pip install datasets")
    ds = load_from_disk(hf_dir)

    tag_names = None
    if "ner_tags" in ds.features and getattr(ds.features["ner_tags"], "feature", None):
        feat = ds.features["ner_tags"].feature
        if hasattr(feat, "names"):
            tag_names = feat.names

    def _get_tokens(ex): return ex.get("tokens") or ex.get("words")
    def _get_boxes(ex):  return ex.get("bboxes") or ex.get("bbox")

    def _get_labels(ex):
        if "labels" in ex and isinstance(ex["labels"], list):
            return ex["labels"]
        if "ner_tags" in ex and tag_names:
            return [tag_names[i] for i in ex["ner_tags"]]
        return None

    for i, ex in enumerate(ds):
        tokens = _get_tokens(ex)
        bboxes = _get_boxes(ex)
        labels = _get_labels(ex)
        if tokens is None or bboxes is None or labels is None:
            continue
        yield {
            "id": str(ex.get("id", f"ex_{i:06d}")),
            "tokens": tokens,
            "bboxes": bboxes,
            "labels": labels,
            "page_ids": ex.get("page_ids", [0] * len(tokens)),
        }


def _normalize_boxes(bboxes: List[List[float]]) -> List[List[int]]:
    """
    Ensure boxes are in 0..1000 grid and x0<=x1, y0<=y1.
    Auto-scale from 0..1 if needed.
    """
    if not bboxes:
        return []

    max_v = max(max(b) for b in bboxes)
    scale = 1000.0 if max_v <= 1.0 else 1.0

    norm = []
    for b in bboxes:
        x0, y0, x1, y1 = [float(v) for v in b]
        x0 = max(0.0, min(1000.0, x0 * scale))
        y0 = max(0.0, min(1000.0, y0 * scale))
        x1 = max(0.0, min(1000.0, x1 * scale))
        y1 = max(0.0, min(1000.0, y1 * scale))
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0
        norm.append([int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))])
    return norm


# ----------------------------- Data types -----------------------------

@dataclass
class Sample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    bbox: torch.Tensor
    labels: torch.Tensor
    page_ids: torch.Tensor
    raw: Dict[str, Any]  # original/normalized record (useful for debugging)


# ----------------------------- Dataset -----------------------------

class FormDataset(Dataset):
    """
    Loader supporting TWO source types:

    - JSON folder:  pass annotations_dir=".../annotations"
    - HF folder:    pass annotations_dir=".../hf_train" (a 'save_to_disk' directory)

    If labels_path is missing or not found, a label map is inferred from the data
    (and 'O' is guaranteed to exist).
    """

    def __init__(
        self,
        annotations_dir: str,
        labels_path: str = "data/labels.json",
        pretrained_name: str = "microsoft/layoutlmv3-base",
        max_length: int = 512,
        use_images: bool = False,
    ):
        self.annotations_dir = annotations_dir
        self.max_length = max_length
        self.use_images = bool(use_images)

        # Load + normalize records
        if _is_hf_folder(annotations_dir):
            records = list(_iter_hf_examples(annotations_dir))
            if not records:
                raise RuntimeError(
                    f"No usable examples found in HF folder: {annotations_dir}\n"
                    "Ensure the dataset has token-level tokens/bboxes/labels."
                )
        else:
            records = list(_iter_json_files(annotations_dir))

        for r in records:
            if "bboxes" in r and isinstance(r["bboxes"], list):
                r["bboxes"] = _normalize_boxes(r["bboxes"])

        self.records: List[Dict[str, Any]] = records

        # Label mapping
        self.label2id: Dict[str, int] = self._load_or_infer_label_map(labels_path)
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Tokenizer/processor
        if self.use_images:
            self.processor = LayoutLMv3Processor.from_pretrained(
                pretrained_name, apply_ocr=False
            )
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_name, use_fast=True
            )
            self.processor = None

    # ---------- label map ----------

    def _load_or_infer_label_map(self, labels_path: str) -> Dict[str, int]:
        if os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            if "O" not in m:
                m = {"O": 0, **{k: (v + 1) for k, v in m.items()}}
            return m

        labels = {"O"}
        for rec in self.records:
            for lbl in rec.get("labels", []):
                labels.add(lbl)
        label2id = {lbl: i for i, lbl in enumerate(sorted(labels))}
        try:
            os.makedirs(os.path.dirname(labels_path) or ".", exist_ok=True)
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(label2id, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return label2id

    # ---------- dunder ----------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Sample:
        ann = self.records[idx]

        words: List[str] = ann["tokens"]
        boxes: List[List[int]] = ann["bboxes"]
        str_labels: List[str] = ann["labels"]
        page_ids_src: List[int] = ann.get("page_ids", [0] * len(words))

        # Map labels → ids (fallback to 'O' if unknown)
        word_label_ids = [self.label2id.get(lbl, self.label2id.get("O", 0)) for lbl in str_labels]

        # --- Encode ---
        if self.tokenizer is not None:
            # TEXT-ONLY path. Do NOT pass is_split_into_words (unsupported in your HF build).
            encoded = self.tokenizer(
                text=words,
                boxes=boxes,                 # LayoutLMv3TokenizerFast supports 'boxes'
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            # IMAGE + TEXT path
            img_path: Optional[str] = ann.get("image_path")
            if not img_path or not os.path.exists(img_path):
                raise RuntimeError("use_images=True but 'image_path' is missing or not found.")
            from PIL import Image
            pil_img = Image.open(img_path).convert("RGB")
            encoded = self.processor(
                images=pil_img,
                text=words,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

        # --- Align labels & page_ids using word_ids() ---
        seq_len = int(encoded["input_ids"].shape[1])
        labels_tensor = torch.full((seq_len,), fill_value=-100, dtype=torch.long)
        page_ids_tensor = torch.zeros((seq_len,), dtype=torch.long)

        # For fast tokenizers, BatchEncoding exposes word_ids()
        # Some versions require batch_index=0.
        try:
            word_ids = encoded.word_ids(batch_index=0)  # type: ignore[attr-defined]
        except TypeError:
            # fallback if no batch_index arg
            word_ids = encoded.word_ids()  # type: ignore[attr-defined]

        if word_ids is None:
            # Fallback: naive trim/pad (shouldn't happen with LayoutLMv3 fast)
            take = min(len(word_label_ids), seq_len)
            labels_tensor[:take] = torch.tensor(word_label_ids[:take], dtype=torch.long)
            pid_take = min(len(page_ids_src), seq_len)
            page_ids_tensor[:pid_take] = torch.tensor(page_ids_src[:pid_take], dtype=torch.long)
        else:
            for i, widx in enumerate(word_ids[:seq_len]):
                if widx is None:
                    continue  # special tokens / padding
                if 0 <= widx < len(word_label_ids):
                    labels_tensor[i] = word_label_ids[widx]
                if 0 <= widx < len(page_ids_src):
                    page_ids_tensor[i] = int(page_ids_src[widx])

        return Sample(
            input_ids=encoded["input_ids"].squeeze(0).long(),
            attention_mask=encoded["attention_mask"].squeeze(0).long(),
            bbox=encoded["bbox"].squeeze(0).long(),
            labels=labels_tensor,
            page_ids=page_ids_tensor,
            raw=ann,
        )


# ----------------------------- Collate -----------------------------

def collate_fn(samples: List[Sample]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([s.input_ids for s in samples]),
        "attention_mask": torch.stack([s.attention_mask for s in samples]),
        "bbox": torch.stack([s.bbox for s in samples]),
        "labels": torch.stack([s.labels for s in samples]),
        "page_ids": torch.stack([s.page_ids for s in samples]),
    }
