# scripts/prelabel_from_model.py
from __future__ import annotations
import os, glob, json, argparse
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer
from utils.field_classifier import FieldClassifier

def normalize_box(b, page_w, page_h):
    x0, y0, x1, y1 = b
    # convert to 0..1000
    return [
        int(round(1000 * max(0, min(1, x0 / page_w)))),
        int(round(1000 * max(0, min(1, y0 / page_h)))),
        int(round(1000 * max(0, min(1, x1 / page_w)))),
        int(round(1000 * max(0, min(1, y1 / page_h)))),
    ]

def extract_tokens_boxes(pdf_path):
    doc = fitz.open(pdf_path)
    out = []
    for i, page in enumerate(doc):
        w, h = page.rect.width, page.rect.height
        words = page.get_text("words")  # list of tuples: x0,y0,x1,y1, word, block_no, line_no, word_no
        words = sorted(words, key=lambda x: (x[3], x[0], x[1]))
        tokens, boxes = [], []
        for x0,y0,x1,y1,word, *_ in words:
            if not word.strip(): continue
            tokens.append(word.strip())
            boxes.append(normalize_box((x0,y0,x1,y1), w, h))
        out.append({"tokens": tokens, "bboxes": boxes, "page_id": i})
    doc.close()
    return out

def word_labels_from_token_logits(tokens, boxes, logits, tokenizer, id2label):
    # map token-level predictions back to word-level using word_ids()
    encoded = tokenizer(
        text=tokens, boxes=boxes, truncation=True, padding="max_length",
        max_length=512, return_tensors="pt"
    )
    with torch.no_grad():
        pred = logits.argmax(-1)[0]  # [T]
    try:
        word_ids = encoded.word_ids(batch_index=0)  # type: ignore
    except TypeError:
        word_ids = encoded.word_ids()               # type: ignore
    word_first = {}
    for i, widx in enumerate(word_ids):
        if widx is None: continue
        if widx not in word_first:
            word_first[widx] = int(pred[i])
    # fill remaining as 'O'
    return [id2label.get(word_first.get(i, 0), "O") for i in range(len(tokens))]

def load_model(checkpoint, backbone, num_labels, device):
    model = FieldClassifier(
        num_labels=num_labels, backbone_name=backbone,
        use_gnn=False, gnn_layers=0, edge_dim=3, dropout=0.1, use_image_branch=False
    ).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser(description="Prelabel PDFs using trained classifier.")
    ap.add_argument("--pdf_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--labels", required=True)              # labels_union.json
    ap.add_argument("--checkpoint", required=True)          # mixed model
    ap.add_argument("--backbone", default="microsoft/layoutlmv3-base")
    ap.add_argument("--device", choices=["cpu","cuda"], default=None)
    args = ap.parse_args()

    with open(args.labels, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {v:k for k,v in label2id.items()}
    num_labels = len(label2id)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    model = load_model(args.checkpoint, args.backbone, num_labels, device)

    os.makedirs(args.out_dir, exist_ok=True)
    pdfs = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
    assert pdfs, f"No PDFs in {args.pdf_dir}"

    for pdf_path in pdfs:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        pages = extract_tokens_boxes(pdf_path)
        for p in pages:
            tokens, boxes = p["tokens"], p["bboxes"]
            if not tokens: continue
            # forward pass
            encoded = tokenizer(
                text=tokens, boxes=boxes, truncation=True, padding="max_length",
                max_length=512, return_tensors="pt"
            )
            batch = {k: v.to(device) for k,v in {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "bbox": encoded["bbox"],
                "labels": torch.full((1,512), -100, dtype=torch.long) # dummy
            }.items()}
            with torch.no_grad():
                out = model(batch, graph=None, labels=None)  # {"logits": [1,T,C]}
                logits = out["logits"].cpu()                 # [1,T,C]

            word_lbls = word_labels_from_token_logits(tokens, boxes, logits, tokenizer, id2label)

            out_rec = {
                "id": f"{base}_p{p['page_id']+1:03d}",
                "tokens": tokens,
                "bboxes": boxes,
                "labels": word_lbls,
                "page_ids": [p["page_id"]] * len(tokens)
            }
            out_path = os.path.join(args.out_dir, f"{base}_p{p['page_id']+1:03d}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out_rec, f, ensure_ascii=False, indent=2)
            print("[ok] wrote", out_path)

if __name__ == "__main__":
    main()
