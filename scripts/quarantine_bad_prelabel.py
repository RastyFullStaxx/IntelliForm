#!/usr/bin/env python3
import os, json, shutil, time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "out"
EMB  = OUT / "llmgnnenhancedembeddings"
OVR  = OUT / "overlay"
GNN  = OUT / "gnn"

SIG = {
  "tokens":[{"text":"Sample","bbox":[20,20,120,50],"page":0}],
  "groups":[]
}

def same_fallback(payload: dict) -> bool:
    try:
        return payload.get("tokens")==SIG["tokens"] and payload.get("groups")==[]
    except Exception:
        return False

def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    Q = OUT / "_quarantine_bad_prelabel" / ts
    Q.mkdir(parents=True, exist_ok=True)
    q_ovr = Q / "overlay"; q_gnn = Q / "gnn"; q_emb = Q / "llmgnnenhancedembeddings"
    for p in (q_ovr, q_gnn, q_emb): p.mkdir(parents=True, exist_ok=True)
    manifest = []

    for jf in EMB.glob("*.json"):
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict): continue
        if not same_fallback(data):   continue

        cid = jf.stem  # file name without .json
        moved = {"cid": cid, "json": None, "overlay": None, "gnn": None}

        # move json
        dst = q_emb / jf.name
        shutil.move(str(jf), str(dst))
        moved["json"] = str(dst.relative_to(Q))

        # move overlay dir if present
        od = OVR / cid
        if od.exists() and od.is_dir():
            dst = q_ovr / cid
            shutil.move(str(od), str(dst))
            moved["overlay"] = str(dst.relative_to(Q))

        # move gnn dir if present
        gd = GNN / cid
        if gd.exists() and gd.is_dir():
            dst = q_gnn / cid
            shutil.move(str(gd), str(dst))
            moved["gnn"] = str(dst.relative_to(Q))

        manifest.append(moved)

    (Q / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Quarantined {len(manifest)} item(s) → {Q}")

if __name__ == "__main__":
    main()
