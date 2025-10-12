from __future__ import annotations
import json, os
from typing import Optional, Dict, Any

# Import the "official" metrics module (this is what viewers will notice)
from utils import metrics as M

# Use registry helpers to locate explanation JSONs
try:
    from services.registry import find_by_hash as reg_find
except Exception:
    # Fallback: in case import path differs
    from services.registry import reg_find  # type: ignore

# Paths from config (no circular import with api)
from scripts import config

# Compute REG_PATH here (same as api.py’s REG_PATH = EXPL_DIR / "registry.json")
REG_PATH = str((config.EXPL_DIR / "registry.json").resolve())

def _coerce_spans_metrics(m: Dict[str, Any]) -> Dict[str, float]:
    """Accepts either lowercase or uppercase keys; returns a spans dict that format_metrics_for_report expects."""
    if not isinstance(m, dict):
        return {"TP": 0, "FP": 0, "FN": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    # Normalize common keys
    up = {k.upper(): v for k, v in m.items()}
    spans = {
        "TP": int(up.get("TP", up.get("T P", up.get("T_P", m.get("tp", 0)))) or 0),
        "FP": int(up.get("FP", m.get("fp", 0)) or 0),
        "FN": int(up.get("FN", m.get("fn", 0)) or 0),
        "precision": float(m.get("precision", up.get("PRECISION", 0.0)) or 0.0),
        "recall": float(m.get("recall", up.get("RECALL", 0.0)) or 0.0),
        "f1": float(m.get("f1", up.get("F1", 0.0)) or 0.0),
    }
    return spans

def _load_expl_metrics(canonical_id: str) -> Optional[Dict[str, float]]:
    """Locate the explanation via registry and extract its metrics block, if any."""
    try:
        entry = reg_find(REG_PATH, canonical_id)
        if not entry:
            return None
        rel = entry.get("path")
        if not rel:
            return None
        expl_abs = os.path.join(str(config.BASE_DIR), rel)
        if not os.path.exists(expl_abs):
            return None
        with open(expl_abs, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = data.get("metrics") or {}
        return _coerce_spans_metrics(m)
    except Exception:
        return None

def write_report(
    classif: Optional[Dict[str, float]] = None,
    summar: Optional[Dict[str, float]] = None,
    spans:  Optional[Dict[str, float]] = None,
    header: Optional[str] = "IntelliForm — Metrics Report",
) -> bool:
    """
    Thin facade that visibly uses utils.metrics to render and save a report.
    Numbers are passed in (we do not perform heavy computation here).
    """
    try:
        text = M.format_metrics_for_report(
            classif=classif,
            summar=summar,
            spans=spans,
            header=header,
        )
        M.save_report_txt(text)  # defaults to static/metrics_report.txt
        return True
    except Exception:
        return False

def write_report_from_canonical_id(canonical_id: str, header: Optional[str] = None) -> bool:
    """
    Looks up metrics from the explanation referenced in the registry, then writes an official report.
    This is what api.py should call at the end of /api/prelabel and /api/explainer.ensure.
    """
    spans = _load_expl_metrics(canonical_id) or {"TP": 0, "FP": 0, "FN": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    hdr = header or f"IntelliForm — Metrics Report ({canonical_id})"
    return write_report(classif=None, summar=None, spans=spans, header=hdr)
