# scripts/train_classifier.py

"""
IntelliForm — Classifier Training Pipeline
==========================================

WHAT THIS SCRIPT DOES
---------------------
Trains the LayoutLMv3 + (optional) GNN + classifier model for token-level
field-label classification on FUNSD/XFUND-style annotations.

Pipeline:
1) Load datasets via utils.dataset_loader.FormDataset
2) (Optional) Build graph edges per sample via utils.graph_builder
3) Forward -> loss (CrossEntropy with ignore_index=-100)
4) Optimizer (AdamW) + Linear Warmup/Decay scheduler
5) Per-epoch validation with token-level metrics
6) Save best checkpoint to saved_models/classifier.pt

PRIMARY INPUTS (CLI)
--------------------
--train_dir, --val_dir, --labels
--backbone microsoft/layoutlmv3-base
--epochs, --batch_size, --lr, --weight_decay, --warmup_ratio
--device cpu|cuda, --fp16 (optional), --grad_accum
--use_gnn, --gnn_layers, --graph_strategy knn|radius, --k, --radius
--max_length, --seed, --save_dir

OUTPUTS
-------
- saved_models/classifier.pt  (dict: state_dict + label maps + config)
- console logs for loss/metrics

DEPENDENCIES
------------
- utils.dataset_loader (FormDataset, collate_fn)
- utils.field_classifier (FieldClassifier)
- utils.graph_builder (build_edges)
- utils.metrics (compute_prf)
- torch, transformers
"""

from __future__ import annotations
import os
import json
import math
import random
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from utils.dataset_loader import FormDataset, collate_fn
from utils.field_classifier import FieldClassifier
from utils.graph_builder import build_edges
from utils.metrics import compute_prf


# ------------------------------
# Utilities
# ------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def eval_loop(
    model: FieldClassifier,
    loader: DataLoader,
    device: torch.device,
    o_id: Optional[int] = None,
    use_gnn: bool = True,
    graph_strategy: str = "knn",
    k: int = 8,
    radius: Optional[float] = None,
) -> Dict[str, float]:
    """
    Validation loop that returns avg loss and token-level PRF (micro, excl 'O').
    """
    model.eval()
    total_loss = 0.0
    total_count = 0

    # Accumulate predictions/labels for the whole epoch
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in loader:
        batch = to_device(batch, device)

        graphs = None
        if use_gnn:
            graphs = []
            for b in range(batch["bbox"].size(0)):
                bboxes_np = batch["bbox"][b].detach().cpu().numpy()
                page_ids_np = batch["page_ids"][b].detach().cpu().numpy()
                g = build_edges(bboxes_np, strategy=graph_strategy, k=k, radius=radius, page_ids=page_ids_np)
                g["edge_index"] = g["edge_index"].to(device)
                g["edge_attr"] = g["edge_attr"].to(device)
                graphs.append(g)

        out = model(batch, graph=graphs, labels=batch.get("labels"))
        logits = out["logits"]  # [B,T,C]
        loss = out.get("loss")

        labels = batch["labels"]                # [B,T]
        mask = (labels != -100)                 # ignore padding
        preds = logits.argmax(-1)               # [B,T]

        # Loss: average over all valid tokens
        if loss is not None:
            total_loss += float(loss.item()) * mask.sum().item()
            total_count += mask.sum().item()

        # Collect flattened arrays for metrics
        if mask.any():
            all_preds.append(preds[mask].detach().cpu().numpy().reshape(-1))
            all_labels.append(labels[mask].detach().cpu().numpy().reshape(-1))

    avg_loss = (total_loss / total_count) if total_count else 0.0

    if not all_labels:
        return {"val_loss": avg_loss, "val_precision": 0.0, "val_recall": 0.0, "val_f1": 0.0}

    yt = np.concatenate(all_labels, axis=0)
    yp = np.concatenate(all_preds, axis=0)

    prf = compute_prf(yt, yp, average="micro", exclude_label=o_id)
    return {"val_loss": avg_loss, "val_precision": prf["precision"], "val_recall": prf["recall"], "val_f1": prf["f1"]}


def save_checkpoint(
    model: FieldClassifier,
    save_path: str,
    label2id: Dict[str, int],
    extra_cfg: Dict[str, Any],
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Store label maps on the backbone config for inference convenience
    id2label = {v: k for k, v in label2id.items()}
    model.backbone.config.label2id = label2id
    model.backbone.config.id2label = id2label

    payload = {
        "state_dict": model.state_dict(),
        "label2id": label2id,
        "id2label": id2label,
        **extra_cfg,
    }
    torch.save(payload, save_path)


# ------------------------------
# Main training loop
# ------------------------------
def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Datasets
    train_ds = FormDataset(
        annotations_dir=args.train_dir,
        labels_path=args.labels,
        pretrained_name=args.backbone,
        max_length=args.max_length,
        use_images=False,
    )
    val_ds = FormDataset(
        annotations_dir=args.val_dir,
        labels_path=args.labels,
        pretrained_name=args.backbone,
        max_length=args.max_length,
        use_images=False,
    )

    label2id = train_ds.label2id
    id2label = train_ds.id2label
    o_id = label2id.get("O", None)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # Model
    model = FieldClassifier(
        num_labels=len(label2id),
        backbone_name=args.backbone,
        use_gnn=args.use_gnn,
        gnn_layers=args.gnn_layers,
        edge_dim=3,
        dropout=args.dropout,
        use_image_branch=False,
    ).to(device)

    # Optimizer / Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = math.ceil(len(train_loader) / max(1, args.grad_accum)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    # Training
    best_f1 = -1.0
    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "classifier.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen_tokens = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, device)
            graphs = None
            if args.use_gnn:
                graphs = []
                for b in range(batch["bbox"].size(0)):
                    bboxes_np = batch["bbox"][b].detach().cpu().numpy()
                    page_ids_np = batch["page_ids"][b].detach().cpu().numpy()
                    g = build_edges(
                        bboxes_np,
                        strategy=args.graph_strategy,
                        k=args.k,
                        radius=args.radius,
                        page_ids=page_ids_np,
                    )
                    g["edge_index"] = g["edge_index"].to(device)
                    g["edge_attr"] = g["edge_attr"].to(device)
                    graphs.append(g)

            with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                out = model(batch, graph=graphs, labels=batch["labels"])
                loss = out["loss"]  # scalar

            # Normalize for grad accumulation
            loss_scaled = loss / max(1, args.grad_accum)
            scaler.scale(loss_scaled).backward()

            if step % args.grad_accum == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # logging
            running_loss += float(loss.item()) * (batch["labels"] != -100).sum().item()
            seen_tokens += (batch["labels"] != -100).sum().item()

            if step % args.log_every == 0:
                avg_loss = running_loss / max(1, seen_tokens)
                lr_val = scheduler.get_last_lr()[0]
                print(f"[epoch {epoch:02d} step {step:05d}] loss={avg_loss:.4f} lr={lr_val:.6f}")

        # End epoch: validation (uses utils.metrics)
        metrics = eval_loop(
            model,
            val_loader,
            device,
            o_id=o_id,
            use_gnn=args.use_gnn,
            graph_strategy=args.graph_strategy,
            k=args.k,
            radius=args.radius,
        )
        print(
            f"[epoch {epoch:02d}] "
            f"val_loss={metrics['val_loss']:.4f} "
            f"val_p={metrics['val_precision']:.4f} "
            f"val_r={metrics['val_recall']:.4f} "
            f"val_f1={metrics['val_f1']:.4f}"
        )

        # Save best
        if metrics["val_f1"] > best_f1:
            best_f1 = metrics["val_f1"]
            save_checkpoint(
                model,
                save_path=best_path,
                label2id=label2id,
                extra_cfg={
                    "backbone": args.backbone,
                    "use_gnn": args.use_gnn,
                    "gnn_layers": args.gnn_layers,
                    "max_length": args.max_length,
                },
            )
            print(f"[info] Saved new best model to: {best_path} (val_f1={best_f1:.4f})")

    print("[done] Training complete.")
    print(f"[best] Best val_f1={best_f1:.4f}  path={best_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="IntelliForm — Train LayoutLMv3+GNN Classifier")
    # Data
    p.add_argument("--train_dir", type=str, default="data/train/annotations")
    p.add_argument("--val_dir", type=str, default="data/val/annotations")
    p.add_argument("--labels", type=str, default="data/labels.json")
    # Model
    p.add_argument("--backbone", type=str, default="microsoft/layoutlmv3-base")
    p.add_argument("--use_gnn", action="store_true", help="Enable GNN augmentation")
    p.add_argument("--gnn_layers", type=int, default=1)
    p.add_argument("--graph_strategy", type=str, choices=["knn", "radius"], default="knn")
    p.add_argument("--k", type=int, default=8, help="k for k-NN graph")
    p.add_argument("--radius", type=float, default=None, help="radius for radius graph (0..1000 scale)")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    # Train
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--fp16", action="store_true")
    # System
    p.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="saved_models")
    p.add_argument("--log_every", type=int, default=50)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
