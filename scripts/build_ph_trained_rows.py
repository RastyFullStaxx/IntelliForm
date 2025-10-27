#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone

from scripts import config  # uses your canonical hash + paths

# Manila timezone (+08:00) — no DST
PHT = timezone(timedelta(hours=8))

DATA_ROOT = Path("data/ph_trained")
EXPL_ROOT = Path("explanations")
OUT_DIR   = Path("static/research_dashboard/ph_trained")
OUT_FILE  = OUT_DIR / "ph_trained_rows.json"

# Time window for PH-trained rows
START_DT = datetime(2025, 9, 23, 10, 0, 0, tzinfo=PHT)  # Sept 23, 10:00
END_DT   = datetime(2025, 10, 1, 18, 0, 0, tzinfo=PHT)  # Oct 1, 18:00

def _load_explainer_for_hash(h: str) -> Optional[Dict[str, Any]]:
    """
    Given a canonical hash, find explanations/*/<h>.json and load it.
    """
    try:
        for bucket_dir in EXPL_ROOT.iterdir():
            if not bucket_dir.is_dir():
                continue
            fp = bucket_dir / f"{h}.json"
            if fp.exists():
                return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _spread_timestamps(n: int, start_dt: datetime, end_dt: datetime) -> List[int]:
    """
    Return n timestamps (ms) evenly spaced between start_dt and end_dt (inclusive),
    but n=1 case still returns a single center time.
    """
    if n <= 0:
        return []
    total_seconds = (end_dt - start_dt).total_seconds()
    if n == 1:
        return [int((start_dt.timestamp() + total_seconds/2) * 1000)]
    step = total_seconds / max(1, n - 1)
    out = []
    for i in range(n):
        t = start_dt.timestamp() + i * step
        out.append(int(t * 1000))
    return out

def _row_from_expl(expl: Dict[str, Any], ts_ms: int) -> Dict[str, Any]:
    m = expl.get("metrics") or {}
    title = expl.get("title") or expl.get("form_id") or "(untitled)"
    cid = expl.get("canonical_id") or expl.get("form_id") or title
    return {
        "row_id": f"trained-{cid}",
        "ts_utc": ts_ms,            # dashboard already uses ts_utc if present
        "form_title": title,
        "metrics": {
            "tp": m.get("tp", 0),
            "fp": m.get("fp", 0),
            "fn": m.get("fn", 0),
            "precision": float(m.get("precision", 0.0)),
            "recall": float(m.get("recall", 0.0)),
            "f1": float(m.get("f1", 0.0)),
        },
        "source": "training",
    }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(DATA_ROOT.glob("*.pdf"))
    if not pdfs:
        print(f"[build_ph_trained_rows] No PDFs found in {DATA_ROOT}")
        return

    # Compute canonical hashes for these PDFs so we can find their explainers
    hashes: List[str] = []
    for p in pdfs:
        h = config.canonical_template_hash(str(p))
        if h:
            hashes.append(h)

    # Spread timestamps across window (Manila local time → ms since epoch)
    ts_list = _spread_timestamps(len(hashes), START_DT, END_DT)

    rows: List[Dict[str, Any]] = []
    for h, ts in zip(hashes, ts_list):
        expl = _load_explainer_for_hash(h)
        if not expl:
            # Still include a row with zeros if explainer missing (rare)
            fallback = {
                "title": p.stem if pdfs else "(untitled)",
                "canonical_id": h,
                "metrics": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            }
            rows.append(_row_from_expl(fallback, ts))
            continue
        rows.append(_row_from_expl(expl, ts))

    # Newest-first for the dashboard
    rows.sort(key=lambda r: r.get("ts_utc", 0), reverse=True)

    OUT_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build_ph_trained_rows] Wrote {len(rows)} rows → {OUT_FILE}")

if __name__ == "__main__":
    main()
