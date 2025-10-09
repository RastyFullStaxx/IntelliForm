# scripts/split_annotations.py
from __future__ import annotations
import os, re, shutil, random, argparse, glob
random.seed(42)

FORM_RE = re.compile(r'^([A-Za-z0-9]+)_[A-Za-z0-9].*')

def form_type_from_name(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    m = FORM_RE.match(base)
    return (m.group(1) if m else "Unknown").upper()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with ALL PH JSON annotations (unsplit).")
    ap.add_argument("--train", required=True, help="dest: data/train/ph_forms/annotations")
    ap.add_argument("--val",   required=True, help="dest: data/val/ph_forms/annotations")
    ap.add_argument("--test",  required=True, help="dest: data/test/ph_forms/annotations")
    ap.add_argument("--ratios", default="0.8,0.1,0.1", help="train,val,test ratios")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="Move instead of copy")
    args = ap.parse_args()
    random.seed(args.seed)

    r_train, r_val, r_test = [float(x) for x in args.ratios.split(",")]
    assert abs((r_train + r_val + r_test) - 1.0) < 1e-6, "ratios must sum to 1"

    files = sorted(glob.glob(os.path.join(args.src, "*.json")))
    assert files, f"No JSON files in {args.src}"

    # group by FormType for stratified split
    groups = {}
    for p in files:
        ft = form_type_from_name(p)
        groups.setdefault(ft, []).append(p)

    for d in (args.train, args.val, args.test):
        os.makedirs(d, exist_ok=True)

    chosen = {"train": [], "val": [], "test": []}
    for ft, lst in groups.items():
        random.shuffle(lst)
        n = len(lst)
        n_tr = int(round(r_train * n))
        n_va = int(round(r_val * n))
        n_te = n - n_tr - n_va
        parts = (lst[:n_tr], lst[n_tr:n_tr+n_va], lst[n_tr+n_va:])
        for name, chunk in zip(("train","val","test"), parts):
            chosen[name].extend(chunk)

    op = shutil.move if args.move else shutil.copy2
    for name, chunk in chosen.items():
        for p in chunk:
            op(p, os.path.join(getattr(args, name), os.path.basename(p)))

    print("Done:", {k: len(v) for k,v in chosen.items()})

if __name__ == "__main__":
    main()
