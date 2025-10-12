#!/usr/bin/env python3
"""
Relocate FUNSD assets into split folders without touching non-FUNSD items.

Target structure:
  out/
    gnn/
      funsd-test/<cid>/page-XX.png
      funsd-train/<cid>/page-XX.png
    overlay/
      funsd-test/<cid>/page-XX.png
      funsd-train/<cid>/page-XX.png
    llmgnnenhancedembeddings/
      funsd-test/<cid>.json
      funsd-train/<cid>.json

What it moves:
  • Any dir named funsd-test-* or funsd-train-* found directly under out/gnn or out/overlay
  • Any funsd-* found under out/overlay/test or out/overlay/train → into overlay/funsd-<split>/
  • Any funsd-*.json under out/llmgnnenhancedembeddings or its train/test subdirs
Nothing else is touched.

Usage:
  python scripts/relocate_funsd_assets.py --dry-run
  python scripts/relocate_funsd_assets.py --commit
"""

from __future__ import annotations
import argparse, json, time, shutil
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "out"
GNN  = OUT / "gnn"
OVR  = OUT / "overlay"
EMB  = OUT / "llmgnnenhancedembeddings"

SPLIT_DIRS = {
    "test":  {"gnn": GNN/"funsd-test",  "overlay": OVR/"funsd-test",  "emb": EMB/"funsd-test"},
    "train": {"gnn": GNN/"funsd-train", "overlay": OVR/"funsd-train", "emb": EMB/"funsd-train"},
}

def is_funsd_name(name: str) -> Tuple[bool, str]:
    """Return (is_funsd, split) where split ∈ {'train','test',''}."""
    n = name.lower()
    if n.startswith("funsd-test-"):
        return True, "test"
    if n.startswith("funsd-train-"):
        return True, "train"
    return False, ""

def record(move_list: List[Tuple[str,str]], src: Path, dst: Path):
    move_list.append((str(src), str(dst)))

def ensure_parents(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def move_dir(move_list: List[Tuple[str,str]], src: Path, dst: Path, dry: bool):
    ensure_parents(dst)
    record(move_list, src, dst)
    if not dry:
        if dst.exists():
            # Merge-copy then remove src to be extra safe
            for item in src.iterdir():
                target = dst / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)
            shutil.rmtree(src)
        else:
            shutil.move(str(src), str(dst))

def move_file(move_list: List[Tuple[str,str]], src: Path, dst: Path, dry: bool):
    ensure_parents(dst)
    record(move_list, src, dst)
    if not dry:
        if dst.exists():
            # keep existing; rename source to avoid clobbering
            dst = dst.with_suffix(dst.suffix + ".dup")
        shutil.move(str(src), str(dst))

def relocate_gnn(move_list: List[Tuple[str,str]], dry: bool):
    if not GNN.exists(): return
    # case A: funsd-* directly under gnn
    for d in GNN.iterdir():
        if not d.is_dir(): continue
        ok, split = is_funsd_name(d.name)
        if ok:
            target = SPLIT_DIRS[split]["gnn"] / d.name
            move_dir(move_list, d, target, dry)
    # case B: gnn/test or gnn/train (rare, but handle)
    for split in ("test","train"):
        sd = GNN / split
        if not sd.exists(): continue
        for d in sd.iterdir():
            if d.is_dir() and d.name.lower().startswith(f"funsd-{split}-"):
                target = SPLIT_DIRS[split]["gnn"] / d.name
                move_dir(move_list, d, target, dry)

def relocate_overlay(move_list: List[Tuple[str,str]], dry: bool):
    if not OVR.exists(): return
    # case A: funsd-* directly under overlay
    for d in OVR.iterdir():
        if not d.is_dir(): continue
        ok, split = is_funsd_name(d.name)
        if ok:
            target = SPLIT_DIRS[split]["overlay"] / d.name
            move_dir(move_list, d, target, dry)
    # case B: overlay/test and overlay/train subfolders containing funsd-*
    for split in ("test","train"):
        sd = OVR / split
        if not sd.exists(): continue
        for d in sd.iterdir():
            if d.is_dir() and d.name.lower().startswith(f"funsd-{split}-"):
                target = SPLIT_DIRS[split]["overlay"] / d.name
                move_dir(move_list, d, target, dry)

def relocate_embeddings(move_list: List[Tuple[str,str]], dry: bool):
    if not EMB.exists(): return
    # case A: funsd-*.json directly under llmgnnenhancedembeddings
    for f in EMB.glob("*.json"):
        n = f.name.lower()
        if n.startswith("funsd-test-"):
            target = SPLIT_DIRS["test"]["emb"] / f.name
            move_file(move_list, f, target, dry)
        elif n.startswith("funsd-train-"):
            target = SPLIT_DIRS["train"]["emb"] / f.name
            move_file(move_list, f, target, dry)
    # case B: jsons already inside train/ or test/
    for split in ("test","train"):
        sd = EMB / split
        if not sd.exists(): continue
        for f in sd.glob("*.json"):
            n = f.name.lower()
            if n.startswith(f"funsd-{split}-"):
                target = SPLIT_DIRS[split]["emb"] / f.name
                move_file(move_list, f, target, dry)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="show moves without changing anything")
    ap.add_argument("--commit", action="store_true", help="perform the moves")
    args = ap.parse_args()
    dry = not args.commit

    # ensure target split dirs exist
    for s in SPLIT_DIRS.values():
        for p in s.values(): p.mkdir(parents=True, exist_ok=True)

    moves: List[Tuple[str,str]] = []
    relocate_gnn(moves, dry)
    relocate_overlay(moves, dry)
    relocate_embeddings(moves, dry)

    ts = time.strftime("%Y%m%d-%H%M%S")
    logdir = OUT / "_relocate_funsd_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": ts,
        "dry_run": dry,
        "moves": [{"src": s, "dst": d} for s, d in moves],
    }
    (logdir / f"relocate_{ts}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"{'(DRY-RUN) ' if dry else ''}Planned moves: {len(moves)}")
    if moves:
        print(f"Manifest: {logdir / ('relocate_'+ts+'.json')}")

if __name__ == "__main__":
    main()
