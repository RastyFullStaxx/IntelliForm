# scripts/dataset_audit.py
from __future__ import annotations
import os, json, glob, argparse
from collections import Counter, defaultdict

def scan_dir(d):
    files = glob.glob(os.path.join(d, "*.json"))
    docs, toks, labels = 0, 0, Counter()
    by_form = Counter()
    for p in files:
        try:
            ex = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
        docs += 1
        tks = ex.get("tokens", [])
        lbs = ex.get("labels", [])
        toks += len(tks)
        labels.update(lbs)
        base = os.path.splitext(os.path.basename(p))[0]
        formtype = base.split("_", 1)[0].upper() if "_" in base else "UNKNOWN"
        by_form[formtype] += 1
    return {"docs": docs, "tokens": toks, "labels": labels, "by_form": by_form}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="comma/semicolon list of annotation dirs")
    ap.add_argument("--labels_map", default="data/labels_union.json")
    args = ap.parse_args()
    dirs = [s.strip() for s in args.dirs.replace(";", ",").split(",") if s.strip()]

    union = {}
    if os.path.exists(args.labels_map):
        union = json.load(open(args.labels_map, "r", encoding="utf-8"))
    union_set = set(union.keys())

    totals = {"docs": 0, "tokens": 0}
    agg_labels = Counter()
    agg_forms = Counter()
    per_dir = {}
    for d in dirs:
        stats = scan_dir(d)
        per_dir[d] = stats
        totals["docs"] += stats["docs"]
        totals["tokens"] += stats["tokens"]
        agg_labels.update(stats["labels"])
        agg_forms.update(stats["by_form"])

    print("\n=== DATASET AUDIT ===")
    for d, s in per_dir.items():
        print(f"{d}: docs={s['docs']}, tokens={s['tokens']}, unique_labels={len(s['labels'])}")

    print(f"\nTOTAL: docs={totals['docs']} tokens={totals['tokens']}")
    print("\nTop 12 labels:", agg_labels.most_common(12))
    print("\nBy FormType:", dict(agg_forms))

    if union:
        missing = sorted([k for k in agg_labels.keys() if k not in union_set])
        if missing:
            print("\n[WARN] Labels present in data but missing in labels_union.json:")
            for m in missing: print("  -", m)
        else:
            print("\n[OK] All labels are present in labels_union.json")

if __name__ == "__main__":
    main()
