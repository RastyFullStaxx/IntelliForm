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

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ------------------------------
# Heavy deps (kept import-level to satisfy module contracts)
# ------------------------------
import torch
import torch.nn.functional as F
from transformers import LayoutLMv3Processor

# IntelliForm stack pieces
from utils.field_classifier import (
    processor as main_processor,   # kept for compatibility (not directly used here)
    embedding as base_embedding,   # exported for other modules to import from here if needed
    model as classifier_model,     # the actual classifier model instance
    FieldClassifier,               # type hint convenience
)
from utils.extractor import extract_pdf
try:
    from utils.graph_builder import build_edges
except Exception:
    build_edges = None  # graph is optional

# T5 summarizer is optional
try:
    from utils.t5_summarize import summarize_fields
    _HAS_T5 = True
except Exception:
    _HAS_T5 = False

# ---------------------------------
# Secondary processor (explicit requirement)
# ---------------------------------
# IMPORTANT: apply_ocr=False (we do not OCR; we use PDF text + layout)
embedding_processor: LayoutLMv3Processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False
)

# Re-export embedding for external imports that expect it here
embedding = base_embedding

# ------------------------------
# Utilities
# ------------------------------
def _write_json(path: str | Path, data: Dict[str, Any]) -> bool:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[llmv3_infer] Failed to write JSON: {e}", file=sys.stderr)
        return False


def _extract_tokens_only(pdf_path: str) -> Dict[str, Any]:
    """
    Extraction for prelabel: returns tokens [+pages], no classification.
    If extraction fails, returns a safe single-token fallback so the pipeline never hard-crashes.
    """
    try:
        ext = extract_pdf(pdf_path)
        toks = []
        for t in getattr(ext, "tokens", []):
            text = str(getattr(t, "text", "")).strip()
            bbox = list(getattr(t, "bbox", []))[:4]
            page = int(getattr(t, "page", 0))
            if text and len(bbox) == 4:
                toks.append({"text": text, "bbox": [int(x) for x in bbox], "page": page})
        if not toks:
            raise ValueError("Extractor returned no tokens.")
        return {"tokens": toks, "groups": []}
    except Exception as e:
        print(f"[llmv3_infer] Extractor fallback for '{pdf_path}' ({e})", file=sys.stderr)
        return {"tokens": [{"text": "Sample", "bbox": [20, 20, 120, 50], "page": 0}], "groups": []}


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
    Collapse token-level BIO sequences into field-level spans.
    Accepts labels like B-NAME / I-NAME / O (configurable if your scheme differs).
    """
    groups: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    def push_cur():
        nonlocal cur
        if cur and cur["tokens"]:
            cur["text"] = " ".join(cur["tokens"])
            groups.append(cur)
        cur = None

    for i, lab in enumerate(labels):
        score = scores[i]
        if score < min_conf:
            push_cur()
            continue

        if lab in ("O", "PAD", None):
            push_cur()
            continue

        if "-" in lab:
            prefix, field = lab.split("-", 1)
        else:
            prefix, field = "B", lab

        tok = tokens[i].strip()
        if not tok:
            continue

        if prefix == "B" or (cur and cur["label"] != field):
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
        else:
            push_cur()
            cur = {
                "label": field,
                "tokens": [tok],
                "bbox": bboxes[i][:],
                "scores": [score],
                "page": page_ids[i] if page_ids else 0
            }

    push_cur()
    for g in groups:
        g["score"] = float(sum(g.get("scores", [0.0])) / max(1, len(g.get("scores", []))))
        g.pop("scores", None)
    return groups


def _encode(tokens: List[str], bboxes: List[List[int]], max_length: int = 512):
    """
    LayoutLMv3 encoding for a single sequence (B=1). No images, apply_ocr=False.
    """
    return embedding_processor(
        text=tokens,
        boxes=bboxes,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ------------------------------
# Public: Prelabel generator (CLI target)
# ------------------------------
def prelabel(pdf_path: str, out_json: str, dev: bool = False) -> bool:
    """
    Generate a temp annotation JSON for a PDF (tokens only; groups left empty).
    The caller (scripts.config) performs TSL duplicate detection and promotion.
    """
    data = _extract_tokens_only(pdf_path)
    ok = _write_json(out_json, data)
    if dev:
        print(f"[llmv3_infer] prelabel → wrote {out_json}: {'OK' if ok else 'FAIL'}")
    return ok


# ------------------------------
# Public: Full inference (optional)
# ------------------------------
def analyze_tokens(
    tokens: List[str],
    bboxes: List[List[int]],
    page_ids: Optional[List[int]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Skips extraction; runs classification, grouping, and summarization.
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

    # Load weights if provided
    if cfg["model_paths"].get("classifier") and os.path.exists(cfg["model_paths"]["classifier"]):
        try:
            sd = torch.load(cfg["model_paths"]["classifier"], map_location="cpu")
            classifier_model.load_state_dict(sd.get("state_dict", sd), strict=False)
        except Exception as e:
            print(f"[llmv3_infer] Warning: could not load classifier weights ({e})", file=sys.stderr)

    classifier_model.to(device).eval()

    # Encode (B=1)
    enc = _encode(tokens, bboxes, max_length=cfg["max_length"])
    batch = _to_device(enc, device)

    # Optional graph
    graph = None
    if cfg.get("graph", {}).get("use", False) and build_edges is not None:
        try:
            bboxes_np = batch["bbox"][0].detach().cpu().numpy()
            graph = build_edges(
                bboxes_np,
                strategy=cfg["graph"].get("strategy", "knn"),
                k=int(cfg["graph"].get("k", 8)),
                radius=cfg["graph"].get("radius", None),
                page_ids=None,
            )
        except Exception as e:
            print(f"[llmv3_infer] Graph build skipped ({e})", file=sys.stderr)
            graph = None

    # Predict
    t0 = time.time()
    with torch.no_grad():
        out = classifier_model(batch, graph=graph, labels=None)  # expects dict with "logits"
        logits = out["logits"]  # [1, T, C]
        probs = F.softmax(logits, dim=-1)
        confs, preds = probs.max(dim=-1)   # [1, T]
        pred_ids = preds[0].tolist()
        conf_vals = [float(x) for x in confs[0].tolist()]
    classify_ms = int((time.time() - t0) * 1000)

    # id2label
    id2label = getattr(classifier_model, "id2label", None)
    if id2label is None and hasattr(getattr(classifier_model, "backbone", None), "config"):
        id2label = getattr(classifier_model.backbone.config, "id2label", None)
    if id2label is None:
        id2label = {i: f"LABEL_{i}" for i in range(max(pred_ids + [0]) + 1)}

    labels = [id2label.get(i, "O") for i in pred_ids]

    # Group BIO spans
    groups = _group_sequences(tokens, bboxes, labels, conf_vals, page_ids=page_ids, min_conf=cfg["min_confidence"])

    # Summarization (optional)
    fields: List[Dict[str, Any]]
    if _HAS_T5 and (cfg["model_paths"].get("t5") is None or os.path.exists(cfg["model_paths"]["t5"])):
        try:
            fields = summarize_fields(groups, cfg["model_paths"].get("t5"))
        except Exception:
            fields = []
    else:
        fields = []

    if not fields:
        # Fallback template summaries
        for g in groups:
            label = g["label"].replace("_", " ").title()
            g["summary"] = f"{label}: {' '.join(g['tokens'])}"
        fields = groups

    # Finalize
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
        "runtime": {"extract_ms": 0, "classify_ms": classify_ms, "summarize_ms": 0},
    }


def analyze_pdf(pdf_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    End-to-end: PDF -> extract -> encode -> classify -> group -> summarize
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    # 1) Extraction
    t0 = time.time()
    try:
        ext = extract_pdf(pdf_path)
        tokens = [str(t.text) for t in getattr(ext, "tokens", [])]
        bboxes = [list(t.bbox) for t in getattr(ext, "tokens", [])]
        page_ids = [int(getattr(t, "page", 0)) for t in getattr(ext, "tokens", [])]
    except Exception as e:
        print(f"[llmv3_infer] Extraction failed in analyze_pdf: {e}", file=sys.stderr)
        tokens, bboxes, page_ids = ["Sample"], [[20, 20, 120, 50]], [0]
    extract_ms = int((time.time() - t0) * 1000)

    # 2..5) Classification + grouping + summarization
    result = analyze_tokens(tokens, bboxes, page_ids=page_ids, config=config or {})
    result["document"] = pdf_path
    result["runtime"]["extract_ms"] = extract_ms
    return result


# ------------------------------
# CLI
# ------------------------------
def _parse_argv(argv: List[str]) -> Dict[str, Any]:
    """
    Minimal arg parser for:
      --prelabel
      --pdf <path>
      --out <temp_json>
      --dev
    Unknown flags are ignored for forward-compatibility.
    """
    out = {"prelabel": False, "pdf": None, "out": None, "dev": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--prelabel":
            out["prelabel"] = True
            i += 1
        elif a == "--pdf" and i + 1 < len(argv):
            out["pdf"] = argv[i + 1]
            i += 2
        elif a == "--out" and i + 1 < len(argv):
            out["out"] = argv[i + 1]
            i += 2
        elif a == "--dev":
            out["dev"] = True
            i += 1
        else:
            i += 1
    return out


def _main(argv: List[str]) -> int:
    args = _parse_argv(argv)

    if args.get("prelabel"):
        pdf = args.get("pdf")
        out_json = args.get("out")
        if not pdf or not out_json:
            print("[llmv3_infer] --prelabel requires --pdf <path> and --out <temp_json>", file=sys.stderr)
            return 2
        ok = prelabel(pdf_path=str(pdf), out_json=str(out_json), dev=bool(args.get("dev")))
        return 0 if ok else 1

    print(
        "Usage:\n"
        "  python -m utils.llmv3_infer --prelabel --pdf <path> --out <temp_json> [--dev]\n",
        file=sys.stderr
    )
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))


# ------------------------------
# Public exports (for clarity)
# ------------------------------
__all__ = [
    "embedding_processor",  # secondary processor (apply_ocr=False)
    "embedding",            # base embedding reference (from utils.field_classifier)
    "prelabel",
    "analyze_tokens",
    "analyze_pdf",
]
