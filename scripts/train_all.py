# scripts/train_all.py

"""
IntelliForm — One-Click Full Training Script
============================================

WHAT THIS SCRIPT DOES
---------------------
Convenience wrapper to sequentially train both IntelliForm components:
1) LayoutLMv3 (+ optional GNN) Classifier   -> scripts/train_classifier.py
2) T5 Summarizer                             -> scripts/train_t5.py

FEATURES
--------
- Shared CLI flags for dataset/model basics
- Optional per-stage overrides
- Graceful skipping of T5 stage if script not found
- Clear logging and error propagation
"""

from __future__ import annotations
import argparse
import os
import sys
import subprocess
from typing import List, Optional

PY = sys.executable  # current Python


def which_file(path: str) -> Optional[str]:
    return path if path and os.path.isfile(path) else None


def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> None:
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80 + "\n")
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(f"[train_all] Subprocess failed with exit code {proc.returncode}: {' '.join(cmd)}")


def main():
    p = argparse.ArgumentParser(description="IntelliForm — Train classifier and (optionally) T5 summarizer")

    # ---- Shared / Classifier defaults ----
    p.add_argument("--train_dir", type=str, default="data/train/annotations")
    p.add_argument("--val_dir", type=str, default="data/val/annotations")
    p.add_argument("--labels", type=str, default="data/labels.json")
    p.add_argument("--backbone", type=str, default="microsoft/layoutlmv3-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--use_gnn", action="store_true")
    p.add_argument("--gnn_layers", type=int, default=1)
    p.add_argument("--graph_strategy", type=str, choices=["knn", "radius"], default="knn")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="saved_models")
    p.add_argument("--log_every", type=int, default=50)

    # ---- Control which stages to run ----
    p.add_argument("--skip_classifier", action="store_true")
    p.add_argument("--skip_t5", action="store_true")

    # ---- T5 stage inputs ----
    p.add_argument("--t5_train_csv", type=str, default="data/train/summaries.csv")
    p.add_argument("--t5_val_csv", type=str, default="data/val/summaries.csv")
    p.add_argument("--t5_model", type=str, default="google/flan-t5-base")
    p.add_argument("--t5_epochs", type=int, default=3)
    p.add_argument("--t5_batch_size", type=int, default=8)
    p.add_argument("--t5_lr", type=float, default=3e-5)
    p.add_argument("--t5_max_input_len", type=int, default=128)
    p.add_argument("--t5_max_target_len", type=int, default=32)
    # ✅ Updated default to a **directory** for HF save/load
    p.add_argument("--t5_save_path", type=str, default="saved_models/t5")

    # ---- Pass-through args ----
    p.add_argument("--extra_classifier_args", type=str, nargs=argparse.REMAINDER)
    p.add_argument("--extra_t5_args", type=str, nargs=argparse.REMAINDER)

    args = p.parse_args()

    clf_script = os.path.join("scripts", "train_classifier.py")
    t5_script = os.path.join("scripts", "train_t5.py")

    # Stage 1: Classifier
    if not args.skip_classifier:
        if not which_file(clf_script):
            raise SystemExit(f"[train_all] Missing {clf_script}.")
        clf_cmd = [
            PY, clf_script,
            "--train_dir", args.train_dir,
            "--val_dir", args.val_dir,
            "--labels", args.labels,
            "--backbone", args.backbone,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--max_length", str(args.max_length),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--warmup_ratio", str(args.warmup_ratio),
            "--grad_accum", str(args.grad_accum),
            "--max_grad_norm", str(args.max_grad_norm),
            "--num_workers", str(args.num_workers),
            "--seed", str(args.seed),
            "--save_dir", args.save_dir,
            "--graph_strategy", args.graph_strategy,
            "--k", str(args.k),
            "--log_every", str(args.log_every),
        ]
        if args.eval_batch_size is not None:
            clf_cmd += ["--eval_batch_size", str(args.eval_batch_size)]
        if args.device is not None:
            clf_cmd += ["--device", args.device]
        if args.fp16:
            clf_cmd += ["--fp16"]
        if args.use_gnn:
            clf_cmd += ["--use_gnn"]
        if args.radius is not None:
            clf_cmd += ["--radius", str(args.radius)]
        if args.dropout is not None:
            clf_cmd += ["--dropout", str(args.dropout)]
        if args.extra_classifier_args:
            clf_cmd += args.extra_classifier_args
        run_cmd(clf_cmd)
    else:
        print("[train_all] Skipping classifier stage (--skip_classifier).")

    # Stage 2: T5
    if not args.skip_t5:
        if not which_file(t5_script):
            print(f"[train_all] {t5_script} not found. Skipping T5 stage.")
        else:
            if not os.path.isfile(args.t5_train_csv):
                print(f"[train_all] WARNING: {args.t5_train_csv} not found.")
            if not os.path.isfile(args.t5_val_csv):
                print(f"[train_all] WARNING: {args.t5_val_csv} not found.")
            t5_cmd = [
                PY, t5_script,
                "--train_csv", args.t5_train_csv,
                "--val_csv", args.t5_val_csv,
                "--model_name", args.t5_model,
                "--epochs", str(args.t5_epochs),
                "--batch_size", str(args.t5_batch_size),
                "--lr", str(args.t5_lr),
                "--max_input_len", str(args.t5_max_input_len),
                "--max_target_len", str(args.t5_max_target_len),
                "--save_path", args.t5_save_path,   # dir by default
            ]
            if args.device is not None:
                t5_cmd += ["--device", args.device]
            if args.fp16:
                t5_cmd += ["--fp16"]
            if args.extra_t5_args:
                t5_cmd += args.extra_t5_args
            run_cmd(t5_cmd)
    else:
        print("[train_all] Skipping T5 stage (--skip_t5).")

    print("\n[train_all] ✅ All requested stages completed.\n")


if __name__ == "__main__":
    main()
