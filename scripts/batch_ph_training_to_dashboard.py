# batch_ph_training_to_dashboard.py
from __future__ import annotations
import sys, json, os, pathlib, argparse, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List
from pathlib import Path
from scripts import config
from services import log_sink

def _is_true(s: str | None) -> bool:
    return str(s or "").strip().lower() in {"1","true","yes","y","on"}

def _load_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def find_training_dir() -> Path:
    expl = config.EXPL_DIR
    if _is_true(os.getenv("INTELLIFORM_STAGING", "0")):
        return expl / "_staging" / "training"
    return expl / "training"

def main():
    ap = argparse.ArgumentParser(description="Backfill PH training explainers into research tool metrics log.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Optional list of filenames (or bare hashes) to include, e.g. 92add0b0...33da.json")
    ap.add_argument("--since", default=None,
                    help="Only include files modified since this ISO timestamp (e.g., 2025-10-12T00:00:00Z)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be logged without writing.")
    ap.add_argument("--source", default="training", help='Source label to use (default: "training")')
    args = ap.parse_args()

    tdir = find_training_dir()
    if not tdir.exists():
        print(f"[backfill] training dir missing: {tdir}")
        return

    only_set = set()
    if args.only:
        for name in args.only:
            name = name.strip()
            if not name:
                continue
            # accept bare hash, or filename
            if not name.endswith(".json") and len(name) >= 16:
                name = f"{name}.json"
            only_set.add(name)

    since_ts = None
    if args.since:
        try:
            since_ts = time.mktime(time.strptime(args.since, "%Y-%m-%dT%H:%M:%SZ"))
        except Exception:
            print(f"[backfill] ignoring invalid --since value: {args.since}")

    files: List[Path] = sorted(tdir.glob("*.json"))
    if only_set:
        files = [p for p in files if p.name in only_set]

    if since_ts is not None:
        files = [p for p in files if p.stat().st_mtime >= since_ts]

    if not files:
        print("[backfill] no explainer JSONs matched.")
        return

    print(f"[backfill] found {len(files)} explainer(s) in {tdir}")

    wrote = 0
    for p in files:
        data = _load_json(p)
        if not data:
            print(f"  - skip (bad json): {p.name}")
            continue

        title = (data.get("title") or data.get("form_id") or p.stem)
        canonical_id = (data.get("canonical_id") or p.stem)
        bucket = (data.get("bucket") or "training")

        m = data.get("metrics") or {}
        row_metrics = {}
        for k in ("tp", "fp", "fn"):
            v = m.get(k)
            try:
                row_metrics[k] = int(v) if v is not None else 0
            except Exception:
                row_metrics[k] = 0
        for k in ("precision","recall","f1"):
            fv = _to_float(m.get(k))
            if fv is not None:
                row_metrics[k] = float(fv)

        row = {
            "canonical_id": canonical_id,
            "form_title": title,
            "bucket": bucket,
            "metrics": row_metrics,
            "source": args.source,                 # "training" for PH training set
            "note": "Backfilled from explainer JSON",
        }

        print(f"  - log: {p.name}  →  {title} [{canonical_id[:12]}…]")
        if not args.dry_run:
            # This writes to the tool-metrics JSONL the dashboard reads
            log_sink.append_tool_metrics(row)
            wrote += 1

    print(f"[backfill] {'would log' if args.dry_run else 'logged'} {wrote} row(s).")

if __name__ == "__main__":
    main()
