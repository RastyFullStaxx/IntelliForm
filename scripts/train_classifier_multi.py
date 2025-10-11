# scripts/train_classifier_multi.py
"""
IntelliForm — Multi-Dataset Trainer (LayoutLMv3 + Classifier)
==============================================================

WHAT THIS DOES
--------------
Trains the FieldClassifier on **multiple datasets at once** (e.g., FUNSD HF + PH JSONs)
using a shared label map (e.g., data/labels_union.json).

WHY THIS SCRIPT (vs train_classifier.py)
---------------------------------------
- Accepts multiple training directories via --train_dirs (comma/semicolon-separated).
- Ensures **label map consistency** across all datasets.
- Optional dataset balancing so small PH sets are not overwhelmed (--balance_datasets).
- Leaves the original single-dataset trainer untouched for reproducible baselines.

INPUTS
------
- --train_dirs : list of dataset dirs to mix (HF save_to_disk or JSON annotations/)
- --val_dir    : single dataset dir for validation
- --labels     : path to a *union* label map (JSON)
- --backbone   : HF model name (default: microsoft/layoutlmv3-base)
- --epochs, --batch_size, --lr, --weight_decay, --num_workers, --fp16, --seed
- --save_dir   : where to save classifier.pt
- --exclude_o  : exclude 'O' from validation metrics (micro PRF)

ASSUMPTIONS
-----------
- All train/val datasets have tokens, bboxes (0..1000 or 0..1), labels compatible with --labels.
- See utils/dataset_loader.py for exact JSON schema.

OUTPUTS
-------
- saved_models/.../classifier.pt
- Console validation metrics per epoch (Precision / Recall / F1)

"""

from __future__ import annotations
import os, math, random, argparse, json
from typing import List
from dataclasses import asdict

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from transformers import get_linear_schedule_with_warmup

from utils.dataset_loader import FormDataset, collate_fn
from utils.field_classifier import FieldClassifier
from utils.metrics import compute_prf

# -------------------- Utils --------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_dirs(arg: str) -> List[str]:
    return [p.strip() for p in arg.replace(";", ",").split(",") if p.strip()]

def same_label_map(ds_list: List[FormDataset]) -> bool:
    base = ds_list[0].label2id
    for ds in ds_list[1:]:
        if ds.label2id != base:
            return False
    return True

def maybe_balance_concat(datasets: List[FormDataset], enable: bool) -> ConcatDataset:
    """
    If enabled, repeats smaller datasets so each contributes roughly equally.
    Simple repetition (no class-wise balancing).
    """
    if not enable or len(datasets) <= 1:
        return ConcatDataset(datasets)
    lens = [len(d) for d in datasets if len(d) > 0]
    if not lens:
        return ConcatDataset(datasets)
    max_len = max(lens)
    balanced_parts = []
    for d in datasets:
        if len(d) == 0:
            continue
        reps = max(1, math.ceil(max_len / len(d)))
        balanced_parts.extend([d] * reps)
    return ConcatDataset(balanced_parts)

# -------------------- Argument Parser --------------------

def build_argparser():
    p = argparse.ArgumentParser(description="IntelliForm — Multi-Dataset Trainer")
    p.add_argument("--train_dirs", type=str, required=True,
                   help="Comma/semicolon-separated list of training dirs (HF or JSON).")
    p.add_argument("--val_dir", type=str, required=True,
                   help="Validation dir (HF or JSON).")
    p.add_argument("--labels", type=str, required=True,
                   help="Path to UNION label map JSON.")
    p.add_argument("--backbone", type=str, default="microsoft/layoutlmv3-base")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--exclude_o", action="store_true", help="Exclude label 'O' from val metrics.")
    p.add_argument("--balance_datasets", action="store_true",
                   help="Repeat small datasets to roughly match the largest.")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p

# -------------------- Training / Eval --------------------

def evaluate(model: FieldClassifier, loader: DataLoader, device: torch.device, o_id: int | None):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(batch, graph=None, labels=None)
            logits = out["logits"]              # [B,T,C]
            preds = logits.argmax(-1)           # [B,T]
            labels = batch["labels"]            # [B,T]
            mask = (labels != -100)
            if mask.any():
                all_preds.append(preds[mask].detach().cpu().numpy().reshape(-1))
                all_labels.append(labels[mask].detach().cpu().numpy().reshape(-1))
    if not all_labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    import numpy as np
    yt = np.concatenate(all_labels, axis=0)
    yp = np.concatenate(all_preds, axis=0)
    return compute_prf(yt, yp, average="micro", exclude_label=o_id)

def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")
    from torch import amp as torch_amp
    scaler = torch_amp.GradScaler('cuda', enabled=use_amp)

    # --------- Datasets ----------
    train_dirs = parse_dirs(args.train_dirs)
    train_parts = [
        FormDataset(
            annotations_dir=td,
            labels_path=args.labels,
            pretrained_name=args.backbone,
            max_length=args.max_length,
            use_images=False,
        ) for td in train_dirs
    ]
    assert all(len(d) > 0 for d in train_parts), "One of the training datasets is empty."

    # All label maps must match (use a union map!)
    assert same_label_map(train_parts), \
        "Label maps differ across train datasets. Use a UNION labels file and re-export."

    # Validation dataset (single)
    val_ds = FormDataset(
        annotations_dir=args.val_dir,
        labels_path=args.labels,
        pretrained_name=args.backbone,
        max_length=args.max_length,
        use_images=False,
    )
    # Also ensure val label map matches
    assert val_ds.label2id == train_parts[0].label2id, \
        "Validation label map differs from train. Use the same UNION labels."

    # Build ConcatDataset (optionally balanced)
    train_ds = maybe_balance_concat(train_parts, enable=args.balance_datasets)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type=="cuda"),
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type=="cuda"),
        collate_fn=collate_fn
    )

    # --------- Model ----------
    num_labels = len(val_ds.label2id)
    model = FieldClassifier(
        num_labels=num_labels,
        backbone_name=args.backbone,
        use_gnn=False, gnn_layers=0, edge_dim=3, dropout=0.1,
        use_image_branch=False,
    ).to(device)

    # Optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr)

    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # for metrics
    o_id = val_ds.label2id.get("O", None) if args.exclude_o else None
    best_f1, best_path = -1.0, os.path.join(args.save_dir, "classifier.pt")

    # --------- Train loop ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch_amp.autocast('cuda', enabled=use_amp):
                out = model(batch, graph=None, labels=batch["labels"])
                loss = out["loss"] if isinstance(out, dict) and "loss" in out else out

            scaler.scale(loss).set_(loss) if hasattr(scaler, "set_") else None  # no-op safeguard
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.detach())

        # ---- Validation ----
        val_metrics = evaluate(model, val_loader, device, o_id=o_id)
        val_p, val_r, val_f1 = val_metrics["precision"], val_metrics["recall"], val_metrics["f1"]
        val_loss = running_loss / max(1, len(train_loader))

        print(f"[epoch {epoch:02d}] val_loss={val_loss:.4f} val_p={val_p:.4f} val_r={val_r:.4f} val_f1={val_f1:.4f}")

        # Save best
        if val_f1 >= best_f1:
            best_f1 = val_f1
            torch.save({"state_dict": model.state_dict(), "args": vars(args)}, best_path)
            print(f"[info] Saved new best model to: {best_path} (val_f1={best_f1:.4f})")

    print("[done] Training complete.")
    print(f"[best] Best val_f1={best_f1:.4f}  path={best_path}")

def main():
    args = build_argparser().parse_args()
    train(args)

if __name__ == "__main__":
    main()
