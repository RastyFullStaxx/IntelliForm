# scripts/train_t5.py

"""
IntelliForm — T5 Summarizer Training Pipeline
=============================================

WHAT THIS SCRIPT DOES
---------------------
Trains a T5 sequence-to-sequence model to generate human-readable summaries
of disambiguated field labels.

It:
1. Loads summarization data from CSVs (train/val).
2. Tokenizes inputs/targets with the T5 tokenizer.
3. Instantiates T5 via utils.t5_summarize.T5Summarizer.
4. Runs the training loop with Teacher Forcing (CrossEntropy over decoder).
5. Evaluates on validation set using ROUGE-L and METEOR (utils.metrics).
6. Saves the best model checkpoint.

CSV SCHEMA (flexible)
---------------------
Accepts any of these column name pairs:
- input|source|text      AND   target|summary|reference

PRIMARY ARGS
------------
--train_csv, --val_csv, --model_name
--epochs, --batch_size, --lr, --device, --fp16
--max_input_len, --max_target_len
--save_path  (directory preferred; .pt allowed but directory will also be saved)

OUTPUTS
-------
- Saved model + tokenizer (preferred): saved_models/t5  (Hugging Face directory)
- Optional: state_dict .pt (if --save_path endswith .pt)

DEPENDENCIES
------------
- utils.t5_summarize
- utils.metrics
- torch, transformers, pandas
"""

from __future__ import annotations
import os
import math
import argparse
import warnings
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from utils.t5_summarize import T5Summarizer
from utils.metrics import compute_rouge_meteor


# ----------------------------
# Dataset
# ----------------------------
class SummarizationCSVDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_input_len: int = 128,
        max_target_len: int = 32,
    ):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}

        # pick flexible column names
        input_col = next((cols[k] for k in ["input", "source", "text"] if k in cols), None)
        target_col = next((cols[k] for k in ["target", "summary", "reference"] if k in cols), None)
        if not input_col or not target_col:
            raise ValueError(
                f"{csv_path} must contain columns like "
                f"[input|source|text] and [target|summary|reference]. Found: {list(df.columns)}"
            )

        self.src_texts = df[input_col].astype(str).tolist()
        self.tgt_texts = df[target_col].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src = self.src_texts[idx]
        tgt = self.tgt_texts[idx]

        enc = self.tokenizer(
            src,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dec = self.tokenizer(
            tgt,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Labels: replace pad_token_id with -100 for CE loss masking
        labels = dec["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }
        return out


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


# ----------------------------
# Training / Eval
# ----------------------------
def train_one_epoch(
    model: T5Summarizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    fp16: bool = False,
    max_grad_norm: float = 1.0,
) -> float:
    model.model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))
    running_loss = 0.0
    running_tokens = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=(fp16 and device.type == "cuda")):
            outputs = model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss  # scalar CE

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        # tokens counted as number of label positions != -100
        valid = (batch["labels"] != -100).sum().item()
        running_loss += float(loss.item()) * max(1, valid)
        running_tokens += max(1, valid)

    return running_loss / max(1, running_tokens)


@torch.no_grad()
def evaluate(
    model: T5Summarizer,
    loader: DataLoader,
    device: torch.device,
    max_target_len: int,
) -> Tuple[float, Dict[str, float]]:
    model.model.eval()

    # Compute NLL loss on val
    running_loss = 0.0
    running_tokens = 0

    # Also gather predictions for ROUGE/METEOR
    refs: List[str] = []
    hyps: List[str] = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        valid = (batch["labels"] != -100).sum().item()
        running_loss += float(loss.item()) * max(1, valid)
        running_tokens += max(1, valid)

        # Generate summaries
        gen_ids = model.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=max_target_len,
            num_beams=4,
            early_stopping=True,
        )
        preds = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        # rebuild references from labels (replace -100 back to pad before decode)
        labels = batch["labels"].clone()
        labels[labels == -100] = model.tokenizer.pad_token_id
        refs_batch = model.tokenizer.batch_decode(labels, skip_special_tokens=True)

        hyps.extend([p.strip() for p in preds])
        refs.extend([r.strip() for r in refs_batch])

    avg_nll = running_loss / max(1, running_tokens)
    # ROUGE/METEOR (gracefully returns zeros if libs missing)
    gen_scores = compute_rouge_meteor(refs, hyps)
    return avg_nll, gen_scores


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="IntelliForm — Train T5 summarizer")
    ap.add_argument("--train_csv", type=str, default="data/train/summaries.csv")
    ap.add_argument("--val_csv", type=str, default="data/val/summaries.csv")
    ap.add_argument("--model_name", type=str, default="google/flan-t5-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_input_len", type=int, default=128)
    ap.add_argument("--max_target_len", type=int, default=32)
    ap.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_path", type=str, default="saved_models/t5")  # directory preferred
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] Using device: {device}")

    # Model
    summarizer = T5Summarizer(model_name_or_path=args.model_name, device=str(device))

    # Data
    train_ds = SummarizationCSVDataset(
        args.train_csv, tokenizer=summarizer.tokenizer,
        max_input_len=args.max_input_len, max_target_len=args.max_target_len
    )
    val_ds = SummarizationCSVDataset(
        args.val_csv, tokenizer=summarizer.tokenizer,
        max_input_len=args.max_input_len, max_target_len=args.max_target_len
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    # Optimizer / Scheduler
    optimizer = torch.optim.AdamW(summarizer.model.parameters(), lr=args.lr)
    total_steps = math.ceil(len(train_loader)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Train
    best_metric = -1.0
    best_dir = args.save_path
    save_dir = args.save_path
    save_pt = None

    # If user provided a .pt path, derive a sibling directory to save full HF weights
    if save_dir.endswith(".pt"):
        base = os.path.splitext(save_dir)[0]
        save_pt = save_dir
        save_dir = base  # use directory for HF save
        warnings.warn(
            f"--save_path ends with .pt; will also save a proper HF directory at '{save_dir}' for inference.",
            UserWarning,
        )

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            summarizer, train_loader, optimizer, scheduler, device, fp16=args.fp16
        )
        val_nll, gen_scores = evaluate(
            summarizer, val_loader, device, max_target_len=args.max_target_len
        )

        rougeL = gen_scores.get("rougeL", 0.0)
        meteor = gen_scores.get("meteor", 0.0)

        print(
            f"[epoch {epoch:02d}] "
            f"train_nll={train_loss:.4f}  val_nll={val_nll:.4f}  "
            f"ROUGE-L={rougeL:.4f}  METEOR={meteor:.4f}"
        )

        # Select best by ROUGE-L primarily (fallback to lowest val_nll if all zeros)
        score = rougeL if rougeL > 0 else (-val_nll)
        if score > best_metric:
            best_metric = score
            # Save HF directory
            summarizer.save_pretrained(save_dir)
            print(f"[info] Saved best T5 to: {save_dir} (score={score:.4f})")
            # Optionally also save .pt state_dict if user asked for it
            if save_pt is not None:
                torch.save(summarizer.model.state_dict(), save_pt)
                print(f"[info] Also saved state_dict to: {save_pt}")

    print("[done] T5 training complete.")
    print(f"[best] Saved to: {save_dir}")
    if save_pt is not None:
        print(f"[best] Also saved state_dict: {save_pt}")


if __name__ == "__main__":
    main()
