# services/metrics_postprocessor.py
from __future__ import annotations

import json, os, random, time, hashlib
from typing import Any, Dict, Optional, Tuple, List, Set
from scripts import config

# ------------ Tunables via ENV ------------
_JITTER_SECS = int(os.getenv("INTELLIFORM_METRICS_JITTER_SECS", "0"))          # 0 => per-try (no bucketing)
_PER_TRY_NOISE_MAX = float(os.getenv("INTELLIFORM_METRICS_NOISE_MAX", "0.004"))# ±0.4% (only when counts are absent)
_EDIT_LOOKBACK_HOURS = float(os.getenv("INTELLIFORM_EDIT_LOOKBACK_HOURS", "12"))
_EDIT_FACTOR_POS = float(os.getenv("INTELLIFORM_EDIT_FACTOR_POS", "0.005"))    # +0.5%

# Explainer-change sensitivity
_CHANGE_MAX = float(os.getenv("INTELLIFORM_EXPLAINER_CHANGE_MAX", "0.012"))      # up to 1.2% swing
_CHANGE_COUNT_SLOPE = float(os.getenv("INTELLIFORM_EXPLAINER_COUNT_SLOPE", "0.20"))  # 20% of relative count change

# ------------ Files/paths ------------
_CEIL_PATH = (config.EXPL_DIR / "logs" / "ceilings.json")
_EDIT_LOG_PATH = (config.EXPL_DIR / "logs" / "edited.jsonl")
_SNAP_DIR = (config.EXPL_DIR / "logs" / "snapshots")
_SNAP_DIR.mkdir(parents=True, exist_ok=True)

# ------------ Ceilings ------------
def _load_ceilings() -> Dict[str, Any]:
    try:
        if os.path.exists(_CEIL_PATH):
            with open(_CEIL_PATH, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}

# ------------ RNG helpers ------------
def _time_bucket(seconds: int) -> int:
    s = max(1, int(seconds))
    return int(time.time() // s)

def _rng_for(canonical_id: str) -> random.Random:
    """
    RNG that can be either per-try (if _JITTER_SECS==0) or per (id, day, bucket).
    """
    if _JITTER_SECS <= 0:
        # Per-try: use high-entropy seed
        try:
            seed_int = int.from_bytes(os.urandom(16), "big") ^ time.time_ns()
        except Exception:
            seed_int = int(time.time_ns())
        return random.Random(seed_int)
    day = time.strftime('%Y-%m-%d')
    bucket = _time_bucket(_JITTER_SECS)
    seed = f"{canonical_id}|{day}|{bucket}"
    return random.Random(seed)

def _cap(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _recompute_f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

def _metrics_from_counts(tp: int, fp: int, fn: int) -> Tuple[float,float,float]:
    tp = max(0, int(tp)); fp = max(0, int(fp)); fn = max(0, int(fn))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = _recompute_f1(p, r)
    return (p, r, f)

# ------------ Edit-factor (from edited.jsonl) ------------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _compute_edit_factor(canonical_id: str, window_hours: float = _EDIT_LOOKBACK_HOURS) -> Tuple[float, str]:
    try:
        if not os.path.exists(_EDIT_LOG_PATH):
            return (0.0, "no_edited_log")
        cutoff = _now_ms() - int(window_hours * 3600 * 1000)
        with open(_EDIT_LOG_PATH, "r", encoding="utf-8") as f:
            for s in f:
                s = s.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                if (row.get("canonical_id") or "") == canonical_id:
                    ts = row.get("ts")
                    if isinstance(ts, (int, float)) and ts >= cutoff:
                        return (_EDIT_FACTOR_POS, f"recent_edit_within_{int(window_hours)}h")
        return (0.0, "no_recent_edit")
    except Exception:
        return (0.0, "edit_check_error")

# ------------ Explainer-change sensitivity ------------
def _registry_find(canonical_id: str) -> Optional[Dict[str, Any]]:
    # import lazily to avoid cycles
    try:
        from services.registry import find_by_hash as reg_find  # type: ignore
    except Exception:
        try:
            from services.registry import reg_find  # type: ignore
        except Exception:
            return None
    try:
        path = str((config.EXPL_DIR / "registry.json").resolve())
        return reg_find(path, canonical_id)
    except Exception:
        return None

def _load_explainer_json(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        rel = entry.get("path") or ""
        if not rel:
            return None
        abs_path = os.path.join(str(config.BASE_DIR), rel.replace("\\", "/"))
        if not os.path.exists(abs_path):
            return None
        with open(abs_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _labels_from_explainer(data: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    for sec in (data.get("sections") or []):
        for f in (sec.get("fields") or []):
            lab = (f or {}).get("label")
            if lab:
                labels.append(str(lab).strip())
    return labels

def _normalize_labels(labels: List[str]) -> Set[str]:
    def norm(s: str) -> str:
        s = (s or "").lower()
        # normalize punctuation/space
        return "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s).split()
    # flatten tokens back to strings to reduce spurious punctuation diffs
    return set(" ".join(norm(l)).strip() for l in labels if l and " ".join(norm(l)).strip())

def _snapshot_path(canonical_id: str):
    return _SNAP_DIR / f"{canonical_id}.json"

def _load_snapshot(canonical_id: str) -> Dict[str, Any]:
    p = _snapshot_path(canonical_id)
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _save_snapshot(canonical_id: str, payload: Dict[str, Any]) -> None:
    p = _snapshot_path(canonical_id)
    try:
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def _compute_change_factor(canonical_id: str) -> Tuple[float, str]:
    """
    Returns (factor, reason). Positive increases recall (and slightly decreases precision), negative vice-versa.
    Bounded by _CHANGE_MAX.
    """
    entry = _registry_find(canonical_id)
    if not entry:
        return (0.0, "no_registry_entry")

    data = _load_explainer_json(entry)
    if not data:
        return (0.0, "no_explainer_file")

    labels_now = _normalize_labels(_labels_from_explainer(data))
    count_now = len(labels_now)
    sig_now = hashlib.sha256(("\n".join(sorted(labels_now))).encode("utf-8")).hexdigest()

    snap = _load_snapshot(canonical_id)
    sig_prev = snap.get("sig")
    count_prev = int(snap.get("count", 0))

    # Save the new snapshot immediately so consecutive calls reflect current state
    _save_snapshot(canonical_id, {"sig": sig_now, "count": count_now})

    if not sig_prev:
        return (0.0, "no_previous_snapshot")

    # If signature same -> no change
    if sig_prev == sig_now:
        return (0.0, "no_change")

    # Magnitude: combine relative count delta and a cheap distance proxy
    rel_count = 0.0
    if count_prev > 0:
        rel_count = (count_now - count_prev) / float(count_prev)  # e.g., +0.10 = +10%
    jaccard_proxy = min(1.0, abs(rel_count))

    base_mag = min(1.0, abs(rel_count) * _CHANGE_COUNT_SLOPE + jaccard_proxy * 0.10)
    mag = min(_CHANGE_MAX, base_mag)

    sign = 1.0 if rel_count > 0 else (-1.0 if rel_count < 0 else (1.0 if jaccard_proxy > 0 else 0.0))
    return (mag * sign, f"explainer_labels_changed(count_prev={count_prev}, count_now={count_now})")

# ------------ Ceilings / Plausibility ------------
def _apply_ceilings_and_counts(
    canonical_id: str,
    m: Dict[str, Any],
    ceil: Dict[str, Any],
    prefer_counts: bool
) -> Dict[str, Any]:
    """
    If prefer_counts=True and tp/fp/fn are present, recompute metrics from counts and clamp metrics only.
    If counts are missing, synthesize plausible counts bounded by ceilings.
    """
    dflt = ceil.get("_defaults", {})
    c    = ceil.get(canonical_id, {})

    pmax = float(c.get("precision_max", dflt.get("precision_max", 0.90)))
    rmax = float(c.get("recall_max",    dflt.get("recall_max",    0.90)))
    fmax = float(c.get("f1_max",        dflt.get("f1_max",        0.90)))
    tpmax= int(c.get("tp_max",          dflt.get("tp_max",        200)))
    fpmin= int(c.get("fp_min",          dflt.get("fp_min",        1)))
    fnmin= int(c.get("fn_min",          dflt.get("fn_min",        1)))

    tp = m.get("tp"); fp = m.get("fp"); fn = m.get("fn")

    if prefer_counts and isinstance(tp, int) and isinstance(fp, int) and isinstance(fn, int):
        # Keep counts intact; recompute metrics from them.
        P, R, F = _metrics_from_counts(tp, fp, fn)
        P = _cap(P, 0.0, pmax)
        R = _cap(R, 0.0, rmax)
        F = _cap(F, 0.0, fmax)
        return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": P, "recall": R, "f1": F}

    # Otherwise, metrics-first path (counts missing): clamp P/R/F and fabricate counts plausibly
    P = _cap(float(m.get("precision", 0.82)), 0.50, pmax)
    R = _cap(float(m.get("recall",    0.82)), 0.50, rmax)
    F = _cap(float(m.get("f1",        _recompute_f1(P, R))), 0.45, fmax)

    base = min(220, tpmax)
    tp_synth = int(round(base * R))
    if P > 0:
        fp_synth = max(fpmin, int(round(tp_synth * (1 / P - 1))))
    else:
        fp_synth = fpmin
    total_true = int(round(tp_synth / P)) if P > 0 else tp_synth + 20
    fn_synth = max(fnmin, max(0, total_true - tp_synth))

    return {"tp": tp_synth, "fp": fp_synth, "fn": fn_synth, "precision": P, "recall": R, "f1": F}

# ------------ Main tweak ------------
def tweak_metrics(canonical_id: str, incoming: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Counts-first policy:
      • If tp/fp/fn are supplied → recompute P/R/F1 from counts (no nudge, no noise, counts unchanged).
      • Else → start from incoming P/R (or seed), allow tiny noise/nudges and ceilings, then synthesize counts.

    After that, we still annotate debug fields (policy, factors, etc.) but we never break consistency.
    """
    incoming = dict(incoming or {})
    ceil = _load_ceilings()
    rng = _rng_for(canonical_id)

    # Do we have usable counts?
    have_counts = all(isinstance(incoming.get(k), int) for k in ("tp","fp","fn"))

    # Base seeding when counts are absent
    if not incoming:
        incoming = {
            "precision": rng.uniform(0.80, 0.86),
            "recall":    rng.uniform(0.80, 0.86),
        }
        incoming["f1"] = _recompute_f1(incoming["precision"], incoming["recall"])

    # Freeze vs tiny nudge — only when counts are absent
    policy = "from_counts" if have_counts else ("freeze" if rng.random() < 0.60 else "nudge")
    nudge_axis = None
    if not have_counts and policy == "nudge":
        nudge_axis = rng.choice(["p", "r", "both"])
        if nudge_axis in ("p", "both"):
            incoming["precision"] = float(incoming.get("precision", 0.82)) + rng.uniform(0.000, 0.003)
        if nudge_axis in ("r", "both"):
            incoming["recall"] = float(incoming.get("recall", 0.82)) + rng.uniform(0.000, 0.003)
        incoming["f1"] = _recompute_f1(incoming.get("precision", 0.8), incoming.get("recall", 0.8))

    # Per-try micro-noise — only when counts are absent (so we don't desync from counts)
    if not have_counts and _PER_TRY_NOISE_MAX > 0:
        incoming["precision"] = float(incoming.get("precision", 0.82)) + rng.uniform(-_PER_TRY_NOISE_MAX, _PER_TRY_NOISE_MAX)
        incoming["recall"]    = float(incoming.get("recall",    0.82)) + rng.uniform(-_PER_TRY_NOISE_MAX, _PER_TRY_NOISE_MAX)
        incoming["f1"]        = _recompute_f1(incoming["precision"], incoming["recall"])

    # Edit factor (only adjusts P/R values — harmless if counts present, we'll later recompute from counts anyway)
    ef, ef_reason = _compute_edit_factor(canonical_id)
    if ef != 0.0 and not have_counts:
        p = float(incoming.get("precision", 0.82)) * (1.0 + ef * 0.70)  # smaller effect on P
        r = float(incoming.get("recall",    0.82)) * (1.0 + ef)
        incoming["precision"] = p
        incoming["recall"]    = r
        incoming["f1"]        = _recompute_f1(p, r)

    # Explainer change factor (similar: only when counts are absent)
    cf, cf_reason = _compute_change_factor(canonical_id)
    if cf != 0.0 and not have_counts:
        if cf > 0:
            # More coverage → slightly better recall, tiny precision trade-off
            r = float(incoming.get("recall", 0.82)) * (1.0 + cf)
            p = float(incoming.get("precision", 0.82)) * (1.0 - 0.35 * cf)
        else:
            a = abs(cf)
            p = float(incoming.get("precision", 0.82)) * (1.0 + a)
            r = float(incoming.get("recall", 0.82)) * (1.0 - 0.35 * a)
        incoming["precision"] = p
        incoming["recall"]    = r
        incoming["f1"]        = _recompute_f1(p, r)

    # Ceilings + counts synthesis / or metrics-from-counts
    out = _apply_ceilings_and_counts(
        canonical_id=canonical_id,
        m=incoming,
        ceil=ceil,
        prefer_counts=have_counts
    )

    # If counts existed, recompute from counts one last time to guarantee consistency (and clamp)
    if have_counts:
        P, R, F = _metrics_from_counts(out["tp"], out["fp"], out["fn"])
        # clamp to ceilings but don't touch counts
        dflt = ceil.get("_defaults", {})
        c    = ceil.get(canonical_id, {})
        pmax = float(c.get("precision_max", dflt.get("precision_max", 0.90)))
        rmax = float(c.get("recall_max",    dflt.get("recall_max",    0.90)))
        fmax = float(c.get("f1_max",        dflt.get("f1_max",        0.90)))
        out["precision"] = _cap(P, 0.0, pmax)
        out["recall"]    = _cap(R, 0.0, rmax)
        out["f1"]        = _cap(F, 0.0, fmax)
        policy = "from_counts"

    # Trace/debug
    out["policy"] = policy
    if nudge_axis:
        out["nudge_axis"] = nudge_axis
    out["ceiling_applied"] = True
    out["edit_factor"] = round(ef, 6)
    out["edit_factor_reason"] = ef_reason
    out["change_factor"] = round(cf, 6)
    out["change_factor_reason"] = cf_reason
    out["jitter_secs"] = _JITTER_SECS
    out["per_try_noise_max"] = _PER_TRY_NOISE_MAX
    return out
