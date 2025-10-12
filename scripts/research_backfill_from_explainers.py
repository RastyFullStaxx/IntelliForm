# scripts/research_backfill_from_explainers.py
from __future__ import annotations
import os, json, argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

from scripts import config
from services import log_sink

def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Backfill tool logs from existing explainer JSONs (no LLM).")
    ap.add_argument("--only", nargs="*", default=None, help="Optional exact filenames (hash.json) to backfill.")
    args = ap.parse_args()

    # Staging vs live
    use_staging = (os.getenv("INTELLIFORM_STAGING","0").lower() in {"1","true","yes"})
    expl_dir = config.EXPL_DIR / ("_staging" if use_staging else "") / "training"

    if not expl_dir.exists():
        print(f"[bf] explainer folder missing: {expl_dir}")
        return

    files: List[Path] = sorted(expl_dir.glob("*.json"))
    if args.only:
        wanted = set(args.only)
        files = [p for p in files if p.name in wanted]

    if not files:
        print("[bf] nothing to backfill.")
        return

    print(f"[bf] scanning {len(files)} explainer file(s) in {expl_dir}")

    ok = 0
    for p in files:
        data = _read_json(p)
        if not data:
            continue

        title = data.get("title") or "(untitled)"
        canonical_id = data.get("canonical_id") or p.stem
        bucket = (data.get("bucket") or "training").lower()
        m = data.get("metrics") or {}

        row = {
            "canonical_id": canonical_id,
            "form_title": title,
            "bucket": bucket,
            "metrics": {k: m.get(k) for k in ("tp","fp","fn","precision","recall","f1") if k in m},
            "source": "training",
            "note": "backfill from existing explainer (no LLM)"
        }

        try:
            log_sink.append_tool_metrics(row)
            ok += 1
            print(f"  [+] {p.name} → logged")
        except Exception as e:
            print(f"  [!] {p.name} → SKIP ({e})")

    print(f"[bf] done. wrote {ok} row(s).")

if __name__ == "__main__":
    main()
