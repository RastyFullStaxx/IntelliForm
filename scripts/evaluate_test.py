# scripts/evaluate_test.py
from __future__ import annotations
import os, argparse, numpy as np, torch
from torch.utils.data import DataLoader

from utils.dataset_loader import FormDataset, collate_fn
from utils.field_classifier import FieldClassifier
from utils.metrics import compute_prf, format_metrics_for_report, save_report_txt

@torch.no_grad()
def evaluate(model: FieldClassifier, loader: DataLoader, device: torch.device, o_id: int | None):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        # Move to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # No GNN during plain eval (set up if you want like in train script)
        out = model(batch, graph=None, labels=None)
        logits = out["logits"]  # [B,T,C]
        preds = logits.argmax(-1)  # [B,T]

        labels = batch["labels"]
        mask = (labels != -100)
        if mask.any():
            all_preds.append(preds[mask].detach().cpu().numpy().reshape(-1))
            all_labels.append(labels[mask].detach().cpu().numpy().reshape(-1))

    if not all_labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    yt = np.concatenate(all_labels, axis=0)
    yp = np.concatenate(all_preds, axis=0)
    return compute_prf(yt, yp, average="micro", exclude_label=o_id)

def build_argparser():
    p = argparse.ArgumentParser(description="IntelliForm — Evaluate on test set")
    p.add_argument("--test_dir", type=str, required=True, help="Path to test data folder (annotations/ or hf_*).")
    p.add_argument("--labels", type=str, default="data/labels.json")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to saved_models/.../classifier.pt")
    p.add_argument("--backbone", type=str, default="microsoft/layoutlmv3-base")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, choices=["cpu","cuda"], default=None)
    p.add_argument("--exclude_o", action="store_true", help="Exclude label 'O' from scoring.")
    p.add_argument("--report_txt", type=str, default="static/metrics_report.txt")
    return p

def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load dataset
    ds = FormDataset(
        annotations_dir=args.test_dir,
        labels_path=args.labels,
        pretrained_name=args.backbone,
        max_length=512,
        use_images=False,
    )
    o_id = ds.label2id.get("O", None) if args.exclude_o else None

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type=="cuda"),
        collate_fn=collate_fn
    )

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    num_labels = len(ds.label2id)
    model = FieldClassifier(
        num_labels=num_labels,
        backbone_name=args.backbone,
        use_gnn=False, gnn_layers=0, edge_dim=3, dropout=0.1,
        use_image_branch=False,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    # Evaluate
    metrics = evaluate(model, loader, device, o_id=o_id)
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall   : {metrics['recall']:.4f}")
    print(f"Test F1       : {metrics['f1']:.4f}")

    # Save report
    report = format_metrics_for_report(classif=metrics, header="IntelliForm — Test Metrics")
    os.makedirs(os.path.dirname(args.report_txt), exist_ok=True)
    save_report_txt(report, args.report_txt)
    print(f"[ok] Wrote report to: {args.report_txt}")

if __name__ == "__main__":
    main()
