from __future__ import annotations
import shutil, json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "out"

STAGING = OUT / "_staging"
STAGING.mkdir(parents=True, exist_ok=True)

MAP = {
  "gnn": OUT / "gnn",
  "overlay": OUT / "overlay",
  "emb": OUT / "llmgnnenhancedembeddings",
}

# destinations inside staging
DEST = {
  "funsd-test": {
    "gnn": STAGING / "gnn" / "funsd-test",
    "overlay": STAGING / "overlay" / "funsd-test",
    "emb": STAGING / "llmgnnenhancedembeddings" / "funsd-test",
  },
  "funsd-train": {
    "gnn": STAGING / "gnn" / "funsd-train",
    "overlay": STAGING / "overlay" / "funsd-train",
    "emb": STAGING / "llmgnnenhancedembeddings" / "funsd-train",
  },
  "ph-supplemental-train": {
    "gnn": STAGING / "gnn" / "ph-supplemental-train",
    "overlay": STAGING / "overlay" / "ph-supplemental-train",
    "emb": STAGING / "llmgnnenhancedembeddings" / "ph-supplemental-train",
  },
}

for d in DEST.values():
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)

def _copy_tree(src: Path, dst: Path):
    # don’t overwrite existing
    if src.is_dir():
        (dst / src.name).mkdir(parents=True, exist_ok=True)
        for q in src.iterdir():
            _copy_tree(q, dst / src.name)
    else:
        outp = dst / src.name
        if not outp.exists():
            shutil.copy2(src, outp)

def main():
    actions = []

    # Rule 1: anything already under funsd-test|train or ph-supplemental-train is copied 1:1 into staging mirrors
    for kind, root in MAP.items():
        if not root.exists():
            continue
        for p in root.iterdir():
            n = p.name.lower()
            if n.startswith("funsd-test"):
                actions.append((p, DEST["funsd-test"][kind]))
            elif n.startswith("funsd-train"):
                actions.append((p, DEST["funsd-train"][kind]))
            elif n.startswith("ph-supplemental-train"):
                actions.append((p, DEST["ph-supplemental-train"][kind]))
            else:
                # Rule 2: hashes/strays → dump into a neutral bucket in staging (no guesswork)
                neutral = STAGING / kind / "_strays"
                neutral.mkdir(parents=True, exist_ok=True)
                actions.append((p, neutral))

    # Execute copies
    log = []
    for src, dst in actions:
        before = len(list(dst.rglob("*")))
        _copy_tree(src, dst)
        after = len(list(dst.rglob("*")))
        log.append({"src": str(src.relative_to(BASE)), "dst": str(dst.relative_to(BASE)), "new_items": max(0, after - before)})

    # Write manifest of what we staged
    manifest = STAGING / "manifest.json"
    manifest.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[staging] wrote {manifest}")

if __name__ == "__main__":
    main()
