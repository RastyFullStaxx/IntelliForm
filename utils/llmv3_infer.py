# utils/llmv3_infer.py

"""
IntelliForm — Unified Inference Engine
======================================

WHAT THIS MODULE DOES
---------------------
Provides a high-level "one call" interface to run the complete IntelliForm
inference pipeline on PDFs or pre-extracted tokens:

1) Extraction (optional): If a PDF path is given, `utils.extractor.extract_pdf`
   yields tokens+bboxes (+pages) without OCR.

2) Classification: Uses LayoutLMv3 (+ optional GNN) via `utils.field_classifier`.
   Optionally loads weights from disk (config["model_paths"]["classifier"]).

3) Grouping: Collapses token-level BIO predictions into field-level spans and
   merges adjacent spans per label.

4) Summarization: Calls `utils.t5_summarize` (if available) to produce concise
   human-ready summaries per field. Falls back to a template if T5 is absent.

5) Packaging: Returns a structured dict for UI/metrics consumers.

PRIMARY ENTRYPOINTS
-------------------
- analyze_pdf(pdf_path: str, config: dict = None) -> dict
- analyze_tokens(tokens: list[str], bboxes: list[list[int]], page_ids: list[int] | None, config: dict = None) -> dict

OUTPUT FORMAT
-------------
{
  "document": "uploads/2025-08-23_form.pdf",
  "fields": [
    {
      "label": "FULL_NAME",
      "score": 0.93,
      "tokens": ["John", "A.", "Doe"],
      "bbox": [x0,y0,x1,y1],
      "summary": "Full legal name detected.",
      "page": 0
    }, ...
  ],
  "runtime": {"extract_ms": 120, "classify_ms": 85, "summarize_ms": 40}
}

CONFIG EXPECTATIONS
-------------------
config = {
  "device": "cuda"|"cpu",
  "min_confidence": 0.0,
  "max_length": 512,
  "graph": { "use": True, "strategy": "knn", "k": 8, "radius": null },
  "model_paths": {
      "classifier": "saved_models/classifier.pt",
      "t5": "saved_models/t5.pt"
  }
}

DEPENDENCIES
------------
- utils.field_classifier : processor, embedding, model, FieldClassifier, predict helper
- utils.graph_builder    : optional graph construction
- utils.extractor        : token/bbox extraction for PDFs
- utils.t5_summarize     : (optional) summarize_fields()

INTERACTIONS
------------
- Called by: inference.py, api.py
- Returns: structured dict for UI or evaluator

NOTES
-----
- Uses LayoutLMv3Processor with apply_ocr=False.
- Groups BIO tags; if your labels are not BIO, tweak `group_sequences`.
- Safe to run with B=1 (single-doc) to avoid graph batching complexity.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, time
import torch
import torch.nn.functional as F

from transformers import LayoutLMv3Processor, LayoutLMv3Model

from utils.extractor import extract_pdf
from utils.field_classifier import processor as main_processor
from utils.field_classifier import embedding as embedding_encoder
from utils.field_classifier import model as classifier_model
from utils.field_classifier import FieldClassifier
from utils.graph_builder import build_edges
# T5 is optional; we handle absence gracefully
try:
    from utils.t5_summarize import summarize_fields
    _HAS_T5 = True
except Exception:
    _HAS_T5 = False


# Secondary processor name expected by your stack
embedding_processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False
)

# Raw encoder exposed (kept for compatibility; same object as imported)
embedding = embedding_encoder


# ------------------------------
# Utilities
# ------------------------------
def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

def _maybe_load_classifier(model: FieldClassifier, path: Optional[str]) -> FieldClassifier:
    if path and os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        state_dict = sd.get("state_dict", sd)
        model.load_state_dict(state_dict, strict=False)
    return model

def _encode(tokens: List[str], bboxes: List[List[int]], max_length: int = 512) -> Dict[str, torch.Tensor]:
    """
    LayoutLMv3 encoding for a single sequence. (No images; apply_ocr=False)
    """
    enc = embedding_processor(
        text=tokens,
        boxes=bboxes,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    return enc

def _merge_bbox(a: List[int], b: List[int]) -> List[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

def _group_sequences(
    tokens: List[str],
    bboxes: List[List[int]],
    labels: List[str],
    scores: List[float],
    page_ids: Optional[List[int]] = None,
    min_conf: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Collapse BIO sequences into field spans.
    Assumes labels like B-NAME, I-NAME, O. Adjust parsing for your schema.
    """
    groups: List[Dict[str, Any]] = []
    cur = None

    def push_cur():
        nonlocal cur
        if cur and cur["tokens"]:
            cur["text"] = " ".join(cur["tokens"])
            groups.append(cur)
        cur = None

    for i, lab in enumerate(labels):
        score = scores[i]
        if score < min_conf:
            # treat as O; close current
            push_cur()
            continue

        if lab == "O" or lab == "PAD":
            push_cur()
            continue

        # split like B-NAME / I-NAME
        if "-" in lab:
            prefix, field = lab.split("-", 1)
        else:
            prefix, field = "B", lab  # non-BIO fallback

        tok = tokens[i].strip()
        if not tok:
            continue

        if prefix == "B" or (cur and cur["label"] != field):
            # start a new span
            push_cur()
            cur = {
                "label": field,
                "tokens": [tok],
                "bbox": bboxes[i][:],
                "scores": [score],
                "page": page_ids[i] if page_ids else 0
            }
        elif prefix == "I" and cur and cur["label"] == field:
            cur["tokens"].append(tok)
            cur["bbox"] = _merge_bbox(cur["bbox"], bboxes[i])
            cur["scores"].append(score)
            # keep first page id as span page
        else:
            # unexpected pattern: start a fresh group
            push_cur()
            cur = {
                "label": field,
                "tokens": [tok],
                "bbox": bboxes[i][:],
                "scores": [score],
                "page": page_ids[i] if page_ids else 0
            }

    push_cur()

    # finalize mean score
    for g in groups:
        if g["scores"]:
            g["score"] = float(sum(g["scores"]) / len(g["scores"]))
        else:
            g["score"] = 0.0
        del g["scores"]

    return groups


def _predict_tokens(
    model: FieldClassifier,
    enc: Dict[str, torch.Tensor],
    device: torch.device,
    graph_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], List[float]]:
    """
    Runs the classifier and returns per-token predicted ids & probabilities.
    """
    batch = {k: v.to(device) for k, v in enc.items()}
    graph = None
    if graph_cfg and graph_cfg.get("use", False):
        # Build edges on-the-fly per sequence (B=1 assumption simplifies graph)
        with torch.no_grad():
            bboxes_np = batch["bbox"][0].cpu().numpy()
            page_ids = None  # could pass if you track them
            strategy = graph_cfg.get("strategy", "knn")
            k = int(graph_cfg.get("k", 8))
            radius = graph_cfg.get("radius", None)
            graph = build_edges(bboxes_np, strategy=strategy, k=k, radius=radius, page_ids=page_ids)

    with torch.no_grad():
        out = model(batch, graph=graph, labels=None)
        logits = out["logits"]  # [1, T, C]
        probs = F.softmax(logits, dim=-1)  # [1, T, C]
        confs, preds = probs.max(dim=-1)   # [1, T]
        pred_ids = preds[0].tolist()
        conf_vals = confs[0].tolist()
    return pred_ids, conf_vals


def _ids_to_labels(pred_ids: List[int], id2label: Dict[int, str]) -> List[str]:
    return [id2label.get(i, "O") for i in pred_ids]


def _summarize(groups: List[Dict[str, Any]], t5_path: Optional[str]) -> List[Dict[str, Any]]:
    if not groups:
        return groups
    # Try T5 if available & path provided; otherwise template fallback
    if _HAS_T5 and (t5_path is None or os.path.exists(t5_path)):
        try:
            return summarize_fields(groups, t5_path)
        except Exception:
            pass
    # Fallback: simple templates
    for g in groups:
        label = g["label"].replace("_", " ").title()
        g["summary"] = f"{label}: {' '.join(g['tokens'])}"
    return groups


# ------------------------------
# Public API
# ------------------------------
def analyze_tokens(
    tokens: List[str],
    bboxes: List[List[int]],
    page_ids: Optional[List[int]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Skips extraction; runs classification, grouping, summarization.
    """
    cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "min_confidence": 0.0,
        "max_length": 512,
        "graph": {"use": True, "strategy": "knn", "k": 8, "radius": None},
        "model_paths": {"classifier": "saved_models/classifier.pt", "t5": "saved_models/t5.pt"},
        **(config or {}),
    }
    device = torch.device(cfg["device"])

    # Load/prepare model
    model = classifier_model
    model = _maybe_load_classifier(model, cfg["model_paths"].get("classifier"))
    model.to(device).eval()

    # Encode
    enc = _encode(tokens, bboxes, max_length=cfg["max_length"])
    # Predict
    t0 = time.time()
    pred_ids, confs = _predict_tokens(model, enc, device, graph_cfg=cfg.get("graph"))
    classify_ms = int((time.time() - t0) * 1000)

    # Map ids->labels (read from processor config if available)
    id2label = getattr(model, "id2label", None)
    if id2label is None:
        # Attempt to fetch from HF config, else assume contiguous integers
        if hasattr(model.backbone.config, "id2label"):
            id2label = model.backbone.config.id2label
        else:
            # fallback generic
            id2label = {i: f"LABEL_{i}" for i in range(max(pred_ids + [0]) + 1)}

    labels = _ids_to_labels(pred_ids, id2label)

    # Group BIO spans
    groups = _group_sequences(tokens, bboxes, labels, confs, page_ids=page_ids, min_conf=cfg["min_confidence"])

    # Summarize
    t1 = time.time()
    fields = _summarize(groups, cfg["model_paths"].get("t5"))
    summarize_ms = int((time.time() - t1) * 1000)

    # Finalize shape
    for f in fields:
        f.setdefault("score", 0.0)
        f.setdefault("summary", "")

    return {
        "document": "tokens",
        "fields": [
            {
                "label": f["label"],
                "score": float(f["score"]),
                "tokens": f["tokens"],
                "bbox": [int(x) for x in f["bbox"]],
                "summary": f.get("summary", ""),
                "page": int(f.get("page", 0)),
            }
            for f in fields
        ],
        "runtime": {"extract_ms": 0, "classify_ms": classify_ms, "summarize_ms": summarize_ms},
    }


def analyze_pdf(pdf_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    End-to-end: PDF -> extract -> encode -> classify -> group -> summarize
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "min_confidence": 0.0,
        "max_length": 512,
        "graph": {"use": True, "strategy": "knn", "k": 8, "radius": None},
        "model_paths": {"classifier": "saved_models/classifier.pt", "t5": "saved_models/t5.pt"},
        **(config or {}),
    }
    device = torch.device(cfg["device"])

    # 1) Extraction
    t0 = time.time()
    ext = extract_pdf(pdf_path)
    extract_ms = int((time.time() - t0) * 1000)

    # Flatten tokens/bboxes/page_ids
    tokens = [t.text for t in ext.tokens]
    bboxes = [list(t.bbox) for t in ext.tokens]
    page_ids = [t.page for t in ext.tokens]

    # 2..5) Reuse analyze_tokens for the rest
    result = analyze_tokens(tokens, bboxes, page_ids=page_ids, config=cfg)
    result["document"] = pdf_path
    result["runtime"]["extract_ms"] = extract_ms
    return result
