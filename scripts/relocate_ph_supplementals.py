# scripts/relocate_ph_supplementals.py
from __future__ import annotations
import os, sys, json, shutil
from pathlib import Path
from typing import Dict, List, Tuple
from scripts import config

"""
Normalize PH supplemental outputs:
- Inputs under uploads/ph-supplemental-forms/{train,test}
- Outputs moved to:
    out/gnn/ph-supplemental-{split}/<hash>/
    out/overlay/ph-supplemental-{split}/<hash>/
    out/llmgnnenhancedembeddings/ph-supplemental-{split}/<hash>.json
- Any misplaced items at:
    out/gnn/<hash>, out/overlay/<hash>, out/llmgnnenhancedembeddings/<hash>.json
  will be relocated if <hash> matches a PDF found in uploads/ph-supplemental-forms/*.
"""

BASE = config.BASE_DIR
UPLOADS = BASE / "uploads" / "ph-supplemental-forms"
OUT = BASE / "out"
LOGDIR = OUT / "_relocate_ph_logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

DESTS = {
    "gnn": OUT / "gnn",
    "overlay": OUT / "overlay",
    "emb": OUT / "llmgnnenhancedembeddings",
}

def _scan_inputs() -> Dict[str, Dict[str, Path]]:
    """
    Return mapping:
      { "train": {hash: pdf_path, ...}, "test": {hash: pdf_path, ...} }
    """
    out: Dict[str, Dict[str, Path]] = {"train": {}, "test": {}}
    for split in ("train", "test"):
        root = UPLOADS / split
        if not root.exists():
            continue
        for p in root.rglob("*.pdf"):
            try:
                h = config.canonical_template_hash(p)
                out[split][h] = p
            except Exception:
                # fallback: use sanitized stem to avoid crashing
                out[split][config.sanitize_form_id(p.name)] = p
    return out

def _ensure_dirs():
    for split in ("train", "test"):
        (DESTS["gnn"] / f"ph-supplemental-{split}").mkdir(parents=True, exist_ok=True)
        (DESTS["overlay"] / f"ph-supplemental-{split}").mkdir(parents=True, exist_ok=True)
        (DESTS["emb"] / f"ph-supplemental-{split}").mkdir(parents=True, exist_ok=True)

def _move_dir(src: Path, dst: Path) -> Tuple[bool, str]:
    if not src.exists():
        return (False, "src_missing")
    if dst.exists():
        return (False, "dst_exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return (True, "moved")

def _move_file(src: Path, dst: Path) -> Tuple[bool, str]:
    if not src.exists():
        return (False, "src_missing")
    if dst.exists():
        return (False, "dst_exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return (True, "moved")

def _relocate_for_hash(h: str, split: str) -> Dict[str, str]:
    """
    Try to relocate any stray outputs matching <hash>.
    """
    result = {}
    # GNN: directory named <hash>
    src_gnn = DESTS["gnn"] / h
    dst_gnn = DESTS["gnn"] / f"ph-supplemental-{split}" / h
    ok, why = _move_dir(src_gnn, dst_gnn)
    result["gnn"] = "ok" if ok else why

    # Overlay: directory named <hash>
    src_ov = DESTS["overlay"] / h
    dst_ov = DESTS["overlay"] / f"ph-supplemental-{split}" / h
    ok, why = _move_dir(src_ov, dst_ov)
    result["overlay"] = "ok" if ok else why

    # Embeddings: file named <hash>.json
    src_emb = DESTS["emb"] / f"{h}.json"
    dst_emb = DESTS["emb"] / f"ph-supplemental-{split}" / f"{h}.json"
    ok, why = _move_file(src_emb, dst_emb)
    result["embeddings"] = "ok" if ok else why

    return result

def main():
    _ensure_dirs()
    mapping = _scan_inputs()  # {split:{hash:pdf}}
    manifest_path = LOGDIR / f"manifest_ph_{config._iso_utc().replace(':','-')}.jsonl"

    moved_count = 0
    total = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for split, hmap in mapping.items():
            for h, pdf in hmap.items():
                total += 1
                res = _relocate_for_hash(h, split)
                row = {
                    "ts": int(__import__("time").time()),
                    "ts_utc": config._iso_utc(),
                    "split": split,
                    "hash": h,
                    "pdf": str(pdf),
                    "moves": res,
                }
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                if any(v == "ok" for v in res.values()):
                    moved_count += 1

    print(f"[relocate_ph] processed={total} moved_any={moved_count} log={manifest_path}")

# if __name__ == "__main__":
#     # Allow optional arg to override uploads root (rare)
#     if len(sys.argv) > 1:
#     #     # Temporary override if needed
#     #     global UPLOADS
#     #     UPLOADS = Path(sys.argv[1]).resolve()
#     # main()
