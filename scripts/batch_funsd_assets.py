#!/usr/bin/env python3
# scripts/batch_funsd_assets.py
"""
Batch-produce facade artifacts for FUNSD:
- overlay PNGs (from services/overlay_renderer.render_overlays)
- GNN visuals (if utils.graph_builder is available)
- enhanced tokens JSON (llmgnnenhancedembeddings)

Flow per item:
  1) Convert FUNSD image -> one-page PDF -> /uploads/funsd/<split>/<id>.pdf
  2) POST /api/prelabel (form_id stub + pdf_disk_path=/uploads/...)
  3) Copy results into split-aware folders:
       out/overlay/<split>/<canonical_id>/
       out/gnn/<split>/<canonical_id>/
       out/llmgnnenhancedembeddings/<split>/<canonical_id>.json
  4) Append to a manifest: out/funsd_manifest_<split>.jsonl

Usage:
  huggingface-cli login   # if needed
  pip install datasets pillow requests tqdm
  python scripts/batch_funsd_assets.py --base-url http://127.0.0.1:8000 --splits train test --limit 0

Notes:
- If GNN visuals are empty, your repo likely lacks utils.graph_builder; overlays + embeddings still work.
- GPU not required for this step.
"""
import argparse, os, io, json, shutil, time, hashlib
from pathlib import Path
from typing import Dict, Any
import requests
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

# ---- paths (relative to repo root) ----
BASE = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE / "uploads" / "funsd"
OUT_DIR = BASE / "out"
OVERLAY_DIR = OUT_DIR / "overlay"
GNN_DIR = OUT_DIR / "gnn"
EMB_DIR = OUT_DIR / "llmgnnenhancedembeddings"

def ensure_dirs():
    for p in (UPLOADS_DIR, OUT_DIR, OVERLAY_DIR, GNN_DIR, EMB_DIR):
        p.mkdir(parents=True, exist_ok=True)

def img_to_pdf_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    # Use RGB to avoid mode issues
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img.save(buf, format="PDF", resolution=300.0)
    return buf.getvalue()

def write_pdf_for_sample(img_arr, dest_pdf: Path):
    # datasets gives a PIL.Image.Image for `image` features
    pil_img = img_arr if isinstance(img_arr, Image.Image) else Image.fromarray(img_arr)
    pdf_bytes = img_to_pdf_bytes(pil_img)
    dest_pdf.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_pdf, "wb") as f:
        f.write(pdf_bytes)

def call_prelabel(base_url: str, form_id: str, pdf_web_path: str) -> Dict[str, Any]:
    # /api/prelabel expects form_id (template hash) and pdf_disk_path (web or abs path under uploads/)
    resp = requests.post(f"{base_url}/api/prelabel", data={
        "form_id": form_id,
        "pdf_disk_path": pdf_web_path,
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()

def copy_tree(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.exists():
        return []
    out = []
    for p in sorted(src_dir.glob("*")):
        if p.is_file():
            tgt = dst_dir / p.name
            shutil.copy2(p, tgt)
            out.append(str(tgt))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if split folders already contain this canonical_id")
    args = ap.parse_args()

    ensure_dirs()

    manifests = {}  # split -> manifest path
    for split in args.splits:
        try:
            dset = load_dataset("nielsr/funsd", split=split)
        except Exception:
            print(f"[warn] split '{split}' not available; skipping.")
            continue

        n = len(dset)
        limit = args.limit if args.limit and args.limit > 0 else n
        print(f"[{split}] processing {limit}/{n} items…")

        man_path = OUT_DIR / f"funsd_manifest_{split}.jsonl"
        manifests[split] = man_path
        fout = open(man_path, "a", encoding="utf-8")

        # split-aware output roots
        split_overlay = OVERLAY_DIR / split
        split_gnn = GNN_DIR / split
        split_emb = EMB_DIR / split
        split_overlay.mkdir(parents=True, exist_ok=True)
        split_gnn.mkdir(parents=True, exist_ok=True)
        split_emb.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(range(limit), ncols=100, desc=f"FASSETS {split}")
        for i in pbar:
            ex = dset[i]
            ex_id = str(ex.get("id", i))
            # Create a stable stub form_id; API will re-hash by template if needed
            form_id_stub = f"funsd-{split}-{ex_id}"
            # 1) write to /uploads/funsd/<split>/<id>.pdf
            dest_pdf = UPLOADS_DIR / split / f"{ex_id}.pdf"
            if not dest_pdf.exists():
                try:
                    write_pdf_for_sample(ex["image"], dest_pdf)
                except Exception as e:
                    pbar.set_postfix_str("img->pdf fail")
                    continue

            # 2) call /api/prelabel
            pdf_web_path = f"/uploads/funsd/{split}/{ex_id}.pdf"  # web path; api will map to disk
            try:
                out = call_prelabel(args.base_url, form_id_stub, pdf_web_path)
            except Exception:
                pbar.set_postfix_str("prelabel fail")
                continue

            canonical_id = out.get("canonical_form_id") or out.get("canonical_id") or form_id_stub

            # Optional: skip if already mirrored
            if args.skip_existing:
                if (split_emb / f"{canonical_id}.json").exists():
                    continue

            # 3) mirror outputs into split folders
            # overlays live in: out/overlay/<canonical_id>/page-*.png
            src_overlay = OVERLAY_DIR / canonical_id
            dst_overlay = split_overlay / canonical_id
            copied_overlay = copy_tree(src_overlay, dst_overlay)

            # gnn lives in: out/gnn/<canonical_id>/page-*.png (may be empty)
            src_gnn = GNN_DIR / canonical_id
            dst_gnn = split_gnn / canonical_id
            copied_gnn = copy_tree(src_gnn, dst_gnn)

            # embeddings JSON lives at: out/llmgnnenhancedembeddings/<canonical_id>.json
            emb_rel = out.get("embeddings_out") or out.get("prelabeled_out") or ""
            copied_emb = ""
            try:
                if emb_rel:
                    emb_abs = (BASE / emb_rel).resolve() if not emb_rel.startswith("/") else (BASE / emb_rel.lstrip("/")).resolve()
                    if not emb_abs.exists():
                        # try the canonical location if API returned a relative path with different root
                        emb_abs = EMB_DIR / f"{canonical_id}.json"
                    if emb_abs.exists():
                        dst_emb = split_emb / f"{canonical_id}.json"
                        dst_emb.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(emb_abs, dst_emb)
                        copied_emb = str(dst_emb)
            except Exception:
                pass

            # 4) manifest entry
            entry = {
                "split": split,
                "id": ex_id,
                "canonical_id": canonical_id,
                "pdf": f"/uploads/funsd/{split}/{ex_id}.pdf",
                "overlay_dir": str(dst_overlay).replace("\\", "/"),
                "gnn_dir": str(dst_gnn).replace("\\", "/"),
                "embedding_json": copied_emb.replace("\\", "/") if copied_emb else "",
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

        fout.close()

    print("Done. Artifacts mirrored by split:")
    for split, man in manifests.items():
        if Path(man).exists():
            print(f" - {split}: {man}")
    print("You can now point reviewers to: out/overlay/<split>/…, out/gnn/<split>/…, out/llmgnnenhancedembeddings/<split>/…")
if __name__ == "__main__":
    main()
