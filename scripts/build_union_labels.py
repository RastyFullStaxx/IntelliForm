# scripts/build_union_labels.py
from __future__ import annotations
import os, re, json, argparse, sys
from typing import List, Set

# Optional HF support (for hf_train/hf_test folders)
try:
    from datasets import load_from_disk  # type: ignore
    _HAS_HF = True
except Exception:
    _HAS_HF = False

BIO_RE = re.compile(r"^(O|[BI]-[A-Z0-9_]+)$")

def parse_dirs(arg: str) -> List[str]:
    return [p.strip() for p in arg.replace(";", ",").split(",") if p.strip()]

def is_hf_folder(p: str) -> bool:
    return os.path.isdir(p) and os.path.exists(os.path.join(p, "dataset_info.json"))

def collect_labels_from_json_dir(d: str) -> Set[str]:
    got = set()
    try:
        names = sorted(os.listdir(d))
    except FileNotFoundError:
        return got
    for name in names:
        if not name.endswith(".json"): 
            continue
        path = os.path.join(d, name)
        try:
            ex = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            continue
        for lb in ex.get("labels", []):
            if isinstance(lb, str): 
                got.add(lb)
    return got

def collect_labels_from_hf(d: str) -> Set[str]:
    if not _HAS_HF:
        print(f"[warn] datasets not installed; skipping HF dir: {d}")
        return set()
    ds = load_from_disk(d)
    # Prefer token-classification names
    if "ner_tags" in ds.features and getattr(ds.features["ner_tags"], "feature", None):
        feat = ds.features["ner_tags"].feature
        names = getattr(feat, "names", None)
        if names: 
            return set(names)
    # Fallback: try 'labels' if present (rare)
    got = set()
    for i, ex in enumerate(ds):
        lbs = ex.get("labels")
        if isinstance(lbs, list):
            got.update([x for x in lbs if isinstance(x, str)])
        if i > 1000 and got:  # don't scan entire set needlessly
            break
    return got

def normalize_bio_set(labels: Set[str]) -> Set[str]:
    out = set()
    for lb in labels:
        if not isinstance(lb, str): 
            continue
        s = lb.strip().upper()
        if BIO_RE.match(s):
            out.add(s)
    return out

def ensure_bi_pairs(labels: Set[str]) -> Set[str]:
    """If B-FOO exists but I-FOO missing (or vice versa), add the missing half."""
    out = set(labels)
    types = {}
    for lb in labels:
        if lb == "O": 
            continue
        tag, typ = lb.split("-", 1)
        types.setdefault(typ, set()).add(tag)
    for typ, tags in types.items():
        if "B" in tags and "I" not in tags: 
            out.add(f"I-{typ}")
        if "I" in tags and "B" not in tags: 
            out.add(f"B-{typ}")
    return out

def load_labels_file(path: str) -> dict:
    if os.path.exists(path):
        m = json.load(open(path, "r", encoding="utf-8"))
        if "O" not in m:
            print("[error] labels file must include 'O'. Aborting.")
            sys.exit(2)
        if m["O"] != 0:
            print(f"[error] 'O' must have id 0 (found {m['O']}). Fix the file manually to avoid ID reshuffles.")
            sys.exit(3)
        return m
    else:
        return {"O": 0}

def main():
    ap = argparse.ArgumentParser(description="Build/Update a union labels JSON safely (append-only).")
    ap.add_argument("--labels", required=True, help="Path to labels_union.json")
    ap.add_argument("--scan_dirs", required=True, help="Comma/semicolon list of dirs (JSON annotations or HF save_to_disk).")
    ap.add_argument("--apply", action="store_true", help="Write changes to --labels (otherwise dry-run).")
    args = ap.parse_args()

    existing = load_labels_file(args.labels)
    found: Set[str] = set()

    for d in parse_dirs(args.scan_dirs):
        if not os.path.exists(d):
            print(f"[warn] missing path: {d}")
            continue

        # 1) Prefer HF detection FIRST
        if is_hf_folder(d):
            got = collect_labels_from_hf(d)
            print(f"[scan] HF {d}: {len(got)} labels")

        # 2) Then plain JSON annotation dirs
        elif os.path.isdir(d) and any(fn.endswith(".json") for fn in os.listdir(d)):
            got = collect_labels_from_json_dir(d)
            print(f"[scan] JSON {d}: {len(got)} labels")

        else:
            print(f"[warn] skipped (not JSON dir or HF folder): {d}")
            got = set()

        found.update(got)

    found = ensure_bi_pairs(normalize_bio_set(found))

    # Determine missing labels (present in data but not in union)
    missing = [lb for lb in sorted(found) if lb not in existing]
    if missing:
        print("\n[diff] New labels to add:")
        next_id = 1 + max(existing.values()) if existing else 0
        for lb in missing:
            print(f"  + {lb}  -> {next_id}")
            next_id += 1
    else:
        print("\n[diff] No new labels to add.")

    if not args.apply:
        print("\n[dry-run] No file was written. Use --apply to update the file.")
        return

    # Apply: append to end, preserve existing IDs
    updated = dict(existing)
    nid = 1 + max(updated.values()) if updated else 0
    for lb in missing:
        updated[lb] = nid
        nid += 1

    os.makedirs(os.path.dirname(args.labels) or ".", exist_ok=True)
    with open(args.labels, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)
    print(f"\n[ok] Wrote updated union labels to: {args.labels}")

if __name__ == "__main__":
    main()
