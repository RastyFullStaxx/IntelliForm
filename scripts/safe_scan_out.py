from __future__ import annotations
import os, json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "out"

def main():
    plan = {"gnn": [], "overlay": [], "emb": []}
    # scatter = items in out/{gnn,overlay,llmgnnenhancedembeddings} that are not under *_/funsd-* or *_/ph-supplemental-*
    roots = {
        "gnn": OUT / "gnn",
        "overlay": OUT / "overlay",
        "emb": OUT / "llmgnnenhancedembeddings",
    }
    for key, root in roots.items():
        if not root.exists():
            continue
        for p in root.iterdir():
            name = p.name.lower()
            if name.startswith(("funsd-test","funsd-train","ph-supplemental-")):
                continue
            if p.is_dir() or (p.is_file() and p.suffix.lower()==".json"):
                plan[key].append(str(p.relative_to(BASE)))

    print(json.dumps(plan, indent=2))

if __name__ == "__main__":
    main()
