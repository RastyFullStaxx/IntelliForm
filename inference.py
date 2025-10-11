# inference.py

"""
IntelliForm — CLI Inference Runner
==================================

WHAT THIS MODULE DOES
---------------------
Runs end-to-end IntelliForm inference *without* the web UI for quick
debugging and batch processing.

PIPELINE STEPS
--------------
1) If --pdf is given:
     - Call `utils.llmv3_infer.analyze_pdf()` → extract → classify → group → summarize
2) If --tokens_json is given:
     - Load pre-extracted tokens/bboxes → `analyze_tokens()`
3) Output:
     - Print JSON to stdout and/or save to --out
     - Optionally write a lightweight metrics report to static/metrics_report.txt

USAGE
-----
python inference.py --pdf "uploads/sample.pdf" \
  --classifier "saved_models/classifier.pt" \
  --t5 "saved_models/t5.pt" \
  --out "outputs/sample_result.json" \
  --device cuda --min_conf 0.15 --save_report

ARGUMENTS
---------
--pdf <path>            Path to input PDF
--tokens_json <path>    Use tokens/bboxes JSON instead of PDF
--classifier <path>     Classifier weights (default: saved_models/classifier.pt)
--t5 <path>             T5 checkpoint (default: saved_models/t5.pt)
--device <cpu|cuda>     Device (auto if omitted)
--min_conf <float>      Confidence threshold (default: 0.0)
--no-graph              Disable graph edges (default: enabled)
--out <path>            Save structured JSON to file
--save_report           Write static/metrics_report.txt with a quick summary

OUTPUT FORMAT
-------------
{
  "document": "...",
  "fields": [
    {"label":"EMAIL","score":0.91,"tokens":["user","@","mail",".","com"],"bbox":[...],"summary":"...","page":0}
  ],
  "runtime": {"extract_ms":..., "classify_ms":..., "summarize_ms":...}
}
"""

from __future__ import annotations
import argparse, json, os, sys, time
from typing import Any, Dict, List

from utils.llmv3_infer import analyze_pdf, analyze_tokens
from utils.metrics import format_metrics_for_report, save_report_txt


def _load_tokens_json(path: str) -> Dict[str, Any]:
    """
    Expected structure:
    {
      "tokens": ["Full", "Name", ":", "John", "Doe"],
      "bboxes": [[x0,y0,x1,y1], ...],   # normalized 0..1000
      "page_ids": [0,0,0,0,0]           # optional
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "tokens" not in data or "bboxes" not in data:
        raise ValueError("tokens_json must contain 'tokens' and 'bboxes'.")
    return data


def _build_config(args: argparse.Namespace) -> Dict[str, Any]:
    device = args.device or ("cuda" if _has_cuda() else "cpu")
    return {
        "device": device,
        "min_confidence": float(args.min_conf),
        "max_length": 512,
        "graph": {
            "use": (not args.no_graph),
            "strategy": "knn",
            "k": 8,
            "radius": None
        },
        "model_paths": {
            "classifier": args.classifier or "saved_models/classifier.pt",
            "t5": args.t5 or "saved_models/t5.pt"
        },
    }


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_report(path: str, result: Dict[str, Any]) -> None:
    fields = result.get("fields", [])
    # Build a minimal "spans" dict for the report (counts only); real IoU needs ground truth
    by_label: Dict[str, int] = {}
    scores = []
    for f in fields:
        by_label[f.get("label","UNK")] = by_label.get(f.get("label","UNK"), 0) + 1
        if "score" in f:
            scores.append(float(f["score"]))
    avg_score = (sum(scores)/len(scores)) if scores else 0.0

    # Compose a friendly text report with runtime info
    rt = result.get("runtime", {})
    header = "IntelliForm — Metrics (Quick Report)"
    text = [
        header,
        "=" * len(header),
        f"Document   : {result.get('document','')}",
        f"Field Count: {len(fields)}",
        "",
        "Counts by Label:",
        *[f"  - {k}: {v}" for k, v in sorted(by_label.items())],
        "",
        f"Average Score: {avg_score:.3f}",
        "",
        "Runtimes (ms):",
        f"  - extract   : {rt.get('extract_ms','-')}",
        f"  - classify  : {rt.get('classify_ms','-')}",
        f"  - summarize : {rt.get('summarize_ms','-')}",
        ""
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description="IntelliForm CLI Inference Runner")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", type=str, help="Path to input PDF")
    src.add_argument("--tokens_json", type=str, help="Path to pre-extracted tokens+bboxes JSON")

    parser.add_argument("--classifier", type=str, default=None, help="Classifier weights path")
    parser.add_argument("--t5", type=str, default=None, help="T5 weights path")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None, help="Device override")
    parser.add_argument("--min_conf", type=float, default=0.0, help="Min confidence threshold")
    parser.add_argument("--no-graph", action="store_true", help="Disable graph edges")
    parser.add_argument("--out", type=str, default=None, help="Save result JSON to this path")
    parser.add_argument("--save_report", action="store_true", help="Write static/metrics_report.txt")

    args = parser.parse_args()
    cfg = _build_config(args)

    try:
        if args.pdf:
            result = analyze_pdf(args.pdf, config=cfg)
        else:
            data = _load_tokens_json(args.tokens_json)
            result = analyze_tokens(
                tokens=data["tokens"],
                bboxes=data["bboxes"],
                page_ids=data.get("page_ids"),
                config=cfg
            )
            # set document field to hint source
            result["document"] = args.tokens_json
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Print to stdout
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Save JSON if requested
    if args.out:
        _write_json(args.out, result)

    # Optionally write the quick metrics report
    if args.save_report:
        report_path = "static/metrics_report.txt"
        _write_report(report_path, result)
        # also echo where it was written
        print(f"\n[info] Metrics report written to: {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
