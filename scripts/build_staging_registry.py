# scripts/build_staging_registry.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, Any

BASE = Path(__file__).resolve().parents[1]
EXPL = BASE / "explanations"
STAGING = EXPL / "_staging"
REG_JSONL = STAGING / "registry.jsonl"
REG_JSON  = STAGING / "registry.json"

def _scan_training_fallback() -> list[Dict[str, Any]]:
    out = []
    training = STAGING / "training"
    if not training.exists():
        return out
    for p in sorted(training.glob("*.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cid = j.get("canonical_id") or p.stem
        row = {
            "canonical_id": cid,
            "title": j.get("title") or p.stem,
            "bucket": j.get("bucket") or "training",
            "source_pdf": j.get("source_pdf") or "",  # may be empty in older JSONs
            "expl_json": str(p.relative_to(BASE)),
            "gnn_dir":  f"out/_staging/gnn/ph-supplemental-train/{cid}",
            "overlay_dir": f"out/_staging/overlay/ph-supplemental-train/{cid}",
            "emb_json": f"out/_staging/llmgnnenhancedembeddings/ph-supplemental-train/{cid}.json",
            "created_at": j.get("created_at"),
            "updated_at": j.get("updated_at"),
            "source": "ph-supplemental",
        }
        out.append(row)
    return out

def main():
    STAGING.mkdir(parents=True, exist_ok=True)
    items = []
    seen = set()
    if REG_JSONL.exists():
        for line in REG_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            cid = j.get("canonical_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            items.append(j)

    # Fallback: if jsonl is empty/missing, scan training directory
    if not items:
        items = _scan_training_fallback()

    payload = {"items": items, "count": len(items)}
    REG_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[staging] wrote {REG_JSON} with {len(items)} item(s)")

if __name__ == "__main__":
    main()
