# utils/metrics.py

"""
IntelliForm — Metrics (Classification + Generation)
===================================================

WHAT THIS MODULE DOES
---------------------
Computes evaluation metrics for:
- Token-level classification (Precision/Recall/F1 — micro & macro)
- Field-level grouping (IoU-based TP/FP/FN → P/R/F1)
- Text generation (ROUGE-L, METEOR) for summaries

WHEN IT'S USED
--------------
- Training/validation: per-epoch token metrics
- Inference/evaluation: final reports (e.g., static/metrics_report.txt)

KEY FUNCTIONS
-------------
- compute_prf(y_true, y_pred, mask=None, average="micro", exclude_label=None) -> dict
- evaluate_spans_iou(pred_spans, gt_spans, iou_threshold=0.5) -> dict
- compute_rouge_meteor(references, hypotheses) -> dict
- format_metrics_for_report(classif: dict|None, summar: dict|None, span: dict|None) -> str
- save_report_txt(text: str, path="static/metrics_report.txt") -> None

DEPENDENCIES
------------
- numpy
- sklearn.metrics (optional; falls back to a simple micro implementation)
- rouge_score (optional)
- nltk (optional; for METEOR)

NOTES
-----
- This module intentionally avoids ECE (Expected Calibration Error).
- All external metrics gracefully degrade if the respective packages are missing.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Sequence, Tuple, Union
import os
import numpy as np

# Optional imports with graceful fallbacks
try:
    from sklearn.metrics import precision_recall_fscore_support
    _HAS_SK = True
except Exception:
    _HAS_SK = False

try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
except Exception:
    _HAS_ROUGE = False

try:
    from nltk.translate.meteor_score import single_meteor_score
    _HAS_METEOR = True
except Exception:
    _HAS_METEOR = False


# ------------------------------
# Helpers
# ------------------------------
_ArrayLike = Union[List[int], np.ndarray]

def _to_np(arr: _ArrayLike) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(arr)

def _apply_mask(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if mask is None:
        return y_true, y_pred
    mask = _to_np(mask).astype(bool)
    return y_true[mask], y_pred[mask]


# ------------------------------
# Token-level PRF
# ------------------------------
def compute_prf(
    y_true: _ArrayLike,
    y_pred: _ArrayLike,
    mask: Optional[_ArrayLike] = None,
    average: str = "micro",            # "micro" or "macro"
    exclude_label: Optional[int] = None,  # e.g., exclude the "O" class from scoring
) -> Dict[str, float]:
    """
    Token-level precision/recall/F1 with optional masking and label exclusion.
    """
    yt = _to_np(y_true).astype(int)
    yp = _to_np(y_pred).astype(int)
    yt, yp = _apply_mask(yt, yp, _to_np(mask) if mask is not None else None)

    if exclude_label is not None:
        keep = yt != exclude_label
        yt, yp = yt[keep], yp[keep]

    if yt.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if _HAS_SK:
        p, r, f1, _ = precision_recall_fscore_support(yt, yp, average=average, zero_division=0)
        return {"precision": float(p), "recall": float(r), "f1": float(f1)}
    else:
        # Minimal micro implementation
        TP = (yt == yp).sum()
        # For micro P/R/F1 we need per-class confusions; as a simple fallback
        # treat micro-F1 ~ accuracy when label distribution is unknown
        acc = float(TP) / float(yt.size)
        return {"precision": acc, "recall": acc, "f1": acc}


# ------------------------------
# Field-level span IoU matching
# ------------------------------
def _iou(boxA: Sequence[int], boxB: Sequence[int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = areaA + areaB - inter
    return float(inter) / float(denom) if denom > 0 else 0.0

def evaluate_spans_iou(
    predicted: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Matches predicted field spans to ground-truth via IoU per label.
    Each item should be {"label": str, "bbox": [x0,y0,x1,y1]}.
    """
    TP = FP = FN = 0
    used = set()

    for p in predicted:
        p_box = p.get("bbox") or p.get("box")
        p_lbl = p.get("label")
        if p_box is None or p_lbl is None:
            FP += 1
            continue
        best_iou, best_j = 0.0, None
        for j, g in enumerate(ground_truth):
            if j in used:
                continue
            if p_lbl != g.get("label"):
                continue
            giou = _iou(p_box, g.get("bbox") or g.get("box"))
            if giou > best_iou:
                best_iou, best_j = giou, j
        if best_iou >= iou_threshold and best_j is not None:
            TP += 1
            used.add(best_j)
        else:
            FP += 1

    FN = len(ground_truth) - len(used)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"TP": TP, "FP": FP, "FN": FN, "precision": precision, "recall": recall, "f1": f1}


# ------------------------------
# Text generation metrics
# ------------------------------
def compute_rouge_meteor(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    ROUGE-L and METEOR over lists of strings. Returns 0.0 if packages are missing.
    """
    n = min(len(references), len(hypotheses))
    if n == 0:
        return {"rougeL": 0.0, "meteor": 0.0}

    refs = references[:n]
    hyps = hypotheses[:n]

    # ROUGE-L
    if _HAS_ROUGE:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_vals = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]
        rougeL = float(np.mean(rouge_vals)) if rouge_vals else 0.0
    else:
        rougeL = 0.0

    # METEOR
    if _HAS_METEOR:
        meteor_vals = [single_meteor_score(r, h) for r, h in zip(refs, hyps)]
        meteor = float(np.mean(meteor_vals)) if meteor_vals else 0.0
    else:
        meteor = 0.0

    return {"rougeL": rougeL, "meteor": meteor}


# ------------------------------
# Reporting
# ------------------------------
def format_metrics_for_report(
    classif: Optional[Dict[str, float]] = None,
    summar: Optional[Dict[str, float]] = None,
    spans: Optional[Dict[str, float]] = None,
    header: Optional[str] = "IntelliForm — Metrics Report"
) -> str:
    lines: List[str] = []
    lines.append(header or "")
    lines.append("=" * len(lines[-1]) if header else "")
    lines.append("")

    if classif is not None:
        lines.append("Token Classification:")
        lines.append(f"  Precision : {classif.get('precision', 0.0):.4f}")
        lines.append(f"  Recall    : {classif.get('recall', 0.0):.4f}")
        lines.append(f"  F1        : {classif.get('f1', 0.0):.4f}")
        lines.append("")

    if spans is not None:
        lines.append("Field Spans (IoU matching):")
        lines.append(f"  TP/FP/FN : {int(spans.get('TP',0))}/{int(spans.get('FP',0))}/{int(spans.get('FN',0))}")
        lines.append(f"  Precision: {spans.get('precision', 0.0):.4f}")
        lines.append(f"  Recall   : {spans.get('recall', 0.0):.4f}")
        lines.append(f"  F1       : {spans.get('f1', 0.0):.4f}")
        lines.append("")

    if summar is not None:
        lines.append("Summary Generation:")
        lines.append(f"  ROUGE-L  : {summar.get('rougeL', 0.0):.4f}")
        lines.append(f"  METEOR   : {summar.get('meteor', 0.0):.4f}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def save_report_txt(text: str, path: str = "static/metrics_report.txt") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
