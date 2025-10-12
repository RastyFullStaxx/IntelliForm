#!/usr/bin/env python3
import os, io, json, shutil
from pathlib import Path
from typing import List
from datasets import load_dataset
from PIL import Image
from services.overlay_renderer import render_overlays, render_gnn_visuals

BASE = Path(__file__).resolve().parent.parent
UPLOADS = BASE / "uploads" / "funsd"
ANN_DIR = BASE / "explanations" / "_annotations"
OUT = BASE / "out"
OVER = OUT / "overlay"
GNN  = OUT / "gnn"
EMB  = OUT / "llmgnnenhancedembeddings"

def ensure_dirs():
    for p in (UPLOADS, ANN_DIR, OVER, GNN, EMB):
        p.mkdir(parents=True, exist_ok=True)

def img_to_pdf_bytes(img: Image.Image) -> bytes:
    if img.mode != "RGB": img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PDF", resolution=300.0)
    return buf.getvalue()

def ensure_pdf(split: str, ex_id: str, image: Image.Image) -> str:
    p = (UPLOADS / split / f"{ex_id}.pdf")
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(img_to_pdf_bytes(image))
    return str(p.resolve())

def write_ann(cid: str, tokens: List[str], bboxes: List[List[int]]) -> str:
    # FUNSD bboxes are 0..1000 already (normalized). Keep them.
    data = {
        "tokens": [
            {"text": t, "bbox": [int(x) for x in bb], "page": 0}
            for t, bb in zip(tokens, bboxes)
        ],
        "groups": []
    }
    p = ANN_DIR / f"{cid}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)

def mirror_to_split(split: str, cid: str):
    # copy overlay & gnn to split subfolders if they exist
    for root in (OVER, GNN):
        src = root / cid
        if src.exists():
            dst = root / split / cid
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copytree(src, dst)
    # copy annotation json to embeddings split
    srcj = ANN_DIR / f"{cid}.json"
    dstj = EMB / split / f"{cid}.json"
    dstj.parent.mkdir(parents=True, exist_ok=True)
    if srcj.exists():
        shutil.copy2(srcj, dstj)

def build_split(split: str):
    ds = load_dataset("nielsr/funsd", split=split)
    mf = OUT / f"funsd_manifest_{split}.jsonl"
    with mf.open("w", encoding="utf-8") as mfw:
        for i, ex in enumerate(ds):
            ex_id = str(ex.get("id", i))
            tokens = ex.get("tokens") or ex.get("words") or []
            bboxes = ex.get("bboxes") or ex.get("boxes") or []
            image  = ex.get("image")
            if not (image and tokens and bboxes and len(tokens)==len(bboxes)):
                continue
            cid = f"funsd-{split}-{ex_id}"
            pdf = ensure_pdf(split, ex_id, image)
            ann = write_ann(cid, tokens, bboxes)
            # Render once to canonical (non-split) location,
            # then mirror to split so we preserve both layouts.
            render_overlays(pdf, ann, str((OVER / cid).resolve()), dpi=180)
            render_gnn_visuals(pdf, ann, str((GNN  / cid).resolve()), strategy="knn", k=8, dpi=180)
            mirror_to_split(split, cid)
            mfw.write(json.dumps({"cid": cid, "split": split, "pdf": pdf}) + "\n")

def main():
    ensure_dirs()
    build_split("train")
    build_split("test")

if __name__ == "__main__":
    main()
