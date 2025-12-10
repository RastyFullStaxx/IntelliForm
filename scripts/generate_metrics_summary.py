"""
Generate a consolidated metrics JSON with per-form and dataset summaries for:
- PH forms (`ph_trained_rows.json`)
- FUNSD test (`funsd_test_rows.json`)
- FUNSD train (`funsd_rows.json`)

Outputs to docs/metrics_summary.json by default.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DASHBOARD = ROOT / "static" / "research_dashboard"


def load_rows(path: Path) -> List[Mapping]:
    data = json.loads(path.read_text())
    return data["rows"] if isinstance(data, dict) else data


def add_iou(metrics: Mapping[str, float]) -> Dict[str, float]:
    tp = metrics.get("tp", 0)
    fp = metrics.get("fp", 0)
    fn = metrics.get("fn", 0)
    denom = tp + fp + fn
    with_iou = dict(metrics)
    with_iou["iou"] = round(tp / denom, 6) if denom else None
    return with_iou


def compute_summary(rows: Sequence[Mapping]) -> Dict[str, object]:
    totals = {
        "tp": sum(r.get("metrics", {}).get("tp", 0) for r in rows),
        "fp": sum(r.get("metrics", {}).get("fp", 0) for r in rows),
        "fn": sum(r.get("metrics", {}).get("fn", 0) for r in rows),
    }
    metric_keys = set()
    for r in rows:
        metric_keys.update(r.get("metrics", {}).keys())
    metric_keys -= {"tp", "fp", "fn"}
    macro: MutableMapping[str, float] = {}
    for key in sorted(metric_keys):
        vals = [r["metrics"][key] for r in rows if key in r.get("metrics", {})]
        if vals:
            macro[key] = round(sum(vals) / len(vals), 6)
    denom = totals["tp"] + totals["fp"] + totals["fn"]
    iou = round(totals["tp"] / denom, 6) if denom else None
    return {
        "count": len(rows),
        "totals": totals,
        "macro": macro,
        "iou": iou,
    }


def build_section(rows: Sequence[Mapping]) -> Dict[str, object]:
    return {
        "summary": compute_summary(rows),
        "forms": [
            {
                "form_no": idx + 1,
                "title": row.get("form_title"),
                "bucket": row.get("bucket"),
                "metrics": add_iou(row.get("metrics", {})),
            }
            for idx, row in enumerate(rows)
        ],
    }


def build_report() -> Dict[str, object]:
    ph_rows = load_rows(RESEARCH_DASHBOARD / "ph_trained" / "ph_trained_rows.json")
    funsd_test_rows = load_rows(RESEARCH_DASHBOARD / "funsd" / "funsd_test_rows.json")
    funsd_train_rows = load_rows(RESEARCH_DASHBOARD / "funsd" / "funsd_rows.json")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ph_forms": build_section(ph_rows),
        "funsd_test": build_section(funsd_test_rows),
        "funsd_train": build_section(funsd_train_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "docs" / "metrics_summary.json",
        help="Output path for consolidated metrics JSON.",
    )
    args = parser.parse_args()

    report = build_report()
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.output} with {report['ph_forms']['summary']['count']} PH forms, "
          f"{report['funsd_test']['summary']['count']} FUNSD test forms, "
          f"{report['funsd_train']['summary']['count']} FUNSD train forms.")


if __name__ == "__main__":
    main()
