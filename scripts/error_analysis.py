# scripts/error_analysis.py
from __future__ import annotations
import os, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from collections import Counter, defaultdict

from utils.dataset_loader import FormDataset, collate_fn
from utils.field_classifier import FieldClassifier
from utils.metrics import compute_prf

@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    ids = []
    for batch in loader:
        ids.extend([s["id"] if isinstance(s, dict) and "id" in s else None for s in getattr(loader.dataset, "records", [])])
        b = {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        out = model(b, graph=None, labels=None)
        preds = out["logits"].argmax(-1)
        mask = (b["labels"] != -100)
        preds_all.append(preds[mask].cpu().numpy().reshape(-1))
        labels_all.append(b["labels"][mask].cpu().numpy().reshape(-1))
    yt = np.concatenate(labels_all, 0)
    yp = np.concatenate(preds_all, 0)
    return yt, yp

def per_label_report(yt, yp, id2label, exclude_o=True):
    labels = sorted(set(yt) | set(yp))
    if exclude_o:
        labels = [l for l in labels if id2label.get(int(l),"O")!="O"]
    rep = []
    for lid in labels:
        tp = int(((yt==lid)&(yp==lid)).sum())
        fp = int(((yt!=lid)&(yp==lid)).sum())
        fn = int(((yt==lid)&(yp!=lid)).sum())
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        rep.append((id2label.get(int(lid), str(lid)), prec, rec, f1, tp, fp, fn))
    rep.sort(key=lambda x: x[3], reverse=True)
    return rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--labels", default="data/labels_union.json")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backbone", default="microsoft/layoutlmv3-base")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    ds = FormDataset(args.test_dir, labels_path=args.labels, pretrained_name=args.backbone, use_images=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    id2label = {v:k for k,v in ds.label2id.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FieldClassifier(num_labels=len(ds.label2id), backbone_name=args.backbone, use_gnn=False, gnn_layers=0, edge_dim=3, dropout=0.1, use_image_branch=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)

    yt, yp = collect_preds(model, loader, device)
    micro = compute_prf(yt, yp, average="micro", exclude_label=ds.label2id.get("O", None))
    print(f"[micro] P={micro['precision']:.4f} R={micro['recall']:.4f} F1={micro['f1']:.4f}")

    report = per_label_report(yt, yp, id2label, exclude_o=True)
    print("\nPer-label (sorted by F1):")
    for name, p, r, f1, tp, fp, fn in report:
        print(f"{name:15s}  P={p:.3f} R={r:.3f} F1={f1:.3f}  TP={tp} FP={fp} FN={fn}")

if __name__ == "__main__":
    main()
