#!/usr/bin/env python3
"""
Batch-generate explainer JSONs (model outputs) for evaluation against reference texts.

Datasets:
  - FUNSD test PDFs: data/funsd/pdfs/test
  - PH supplemental forms: data/ph_trained/ph-supplemental-forms

Outputs:
  - static/research_dashboard/funsd/explanations/<canonical_id>.json
  - static/research_dashboard/ph_trained/explanations/<canonical_id>.json

Uses the IntelliForm explainer prompts in scripts/config.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts import config

# Paths
BASE = config.BASE_DIR
FUNSD_PDF_DIR = BASE / "data" / "funsd" / "pdfs" / "test"
PH_PDF_DIR = BASE / "data" / "ph_trained" / "ph-supplemental-forms"

OUT_FUNSD = BASE / "static" / "research_dashboard" / "funsd" / "explanations"
OUT_PH = BASE / "static" / "research_dashboard" / "ph_trained" / "explanations"
REF_FUNSD = BASE / "static" / "research_dashboard" / "funsd" / "references"
REF_PH = BASE / "static" / "research_dashboard" / "ph_trained" / "references"

MAX_SNIPPET_CHARS = int(os.getenv("INTELLIFORM_EXPL_SNIPPET_MAX_CHARS", "20000") or 20000)
MAX_TOKENS = int(os.getenv("INTELLIFORM_ENGINE_MAXTOK", os.getenv("INTELLIFORM_MAX_TOKENS", "16000")) or 16000)


# Helpers
def _iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_write(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _title_from_name(path: Path) -> str:
    s = path.stem.replace("_", " ").replace("-", " ")
    return " ".join(s.split()).strip()


def _bucket_from_ph_name(path: Path) -> str:
    stem = path.stem.lower()
    for key in ("government", "banking", "finance", "health", "healthcare", "insurance", "tax"):
        if key in stem:
            return "government" if key == "government" else key
    return "ph-supplemental"


def _load_reference_labels(ref_path: Path) -> Optional[List[str]]:
    if not ref_path.exists():
        return None
    try:
        data = json.loads(ref_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    labs = []
    # pull from sections->fields and flattened fields
    for sec in data.get("sections") or []:
        for f in sec.get("fields") or []:
            lab = (f or {}).get("label")
            if lab:
                labs.append(str(lab))
    for f in data.get("fields") or []:
        lab = (f or {}).get("label")
        if lab:
            labs.append(str(lab))
    seen = set()
    out = []
    for l in labs:
        k = l.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(l.strip())
    return out or None


def _build_messages_with_labels(
    *,
    canonical_id: str,
    bucket_guess: str,
    title_guess: str,
    text_snippet: str,
    candidate_labels: Optional[List[str]],
) -> List[Dict[str, Any]]:
    return config.build_explainer_messages_with_context(
        canonical_id=canonical_id,
        bucket_guess=bucket_guess,
        title_guess=title_guess,
        text_snippet=text_snippet,
        candidate_labels=candidate_labels,
    )


def _chat_explainer(messages: List[Dict[str, Any]], model: Optional[str], temperature: Optional[float]) -> str:
    return config.chat_completion(
        model=model or config.ENGINE_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=config.TEMPERATURE if temperature is None else float(temperature),
        enforce_json=True,
    )


def _coerce_payload(
    raw: str,
    *,
    canonical_id: str,
    bucket: str,
    title_guess: str,
    source_pdf: Path,
) -> Dict[str, Any]:
    try:
        data = json.loads((raw or "").strip())
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    now = _iso()
    data.setdefault("title", title_guess or canonical_id)
    data.setdefault("form_id", config.sanitize_form_id(title_guess or canonical_id))
    data["canonical_id"] = canonical_id
    data["bucket"] = bucket
    data.setdefault("schema_version", 1)
    data.setdefault("created_at", now)
    data["updated_at"] = now
    data.setdefault("aliases", [])
    data["source_pdf"] = str(source_pdf.relative_to(BASE))
    # ensure sections/fields exist
    sections = data.get("sections")
    if not isinstance(sections, list):
        data["sections"] = []
    fields = data.get("fields")
    if not isinstance(fields, list):
        data["fields"] = []
    return data


def _process_pdf(
    pdf_path: Path,
    *,
    dataset: str,
    out_dir: Path,
    ref_dir: Path,
    bucket: str,
    overwrite: bool,
    model: Optional[str],
    temperature: Optional[float],
) -> Path:
    canonical_id = config.canonical_template_hash(pdf_path)
    out_path = out_dir / f"{canonical_id}.json"
    if out_path.exists() and not overwrite:
        return out_path

    # inputs
    title_guess = _title_from_name(pdf_path)
    snippet = config.quick_text_snippet(str(pdf_path), max_chars=MAX_SNIPPET_CHARS)

    ref_labels = _load_reference_labels(ref_dir / f"{canonical_id}.json")
    messages = _build_messages_with_labels(
        canonical_id=canonical_id,
        bucket_guess=bucket,
        title_guess=title_guess,
        text_snippet=snippet,
        candidate_labels=ref_labels,
    )

    raw = _chat_explainer(messages, model=model, temperature=temperature)
    payload = _coerce_payload(
        raw,
        canonical_id=canonical_id,
        bucket=bucket,
        title_guess=title_guess,
        source_pdf=pdf_path,
    )
    _safe_write(out_path, payload)
    return out_path


def _process_via_reference(
    ref_path: Path,
    *,
    out_dir: Path,
    overwrite: bool,
    model: Optional[str],
    temperature: Optional[float],
) -> Optional[Path]:
    """Preferred for PH: use canonical_id + source_pdf from reference to avoid hash drift."""
    try:
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    canonical_id = str(ref.get("canonical_id") or "").strip()
    source_pdf = ref.get("source_pdf") or ""
    if not canonical_id or not source_pdf:
        return None
    pdf_path = (BASE / source_pdf.replace("\\", "/")).resolve()
    if not pdf_path.exists():
        return None
    bucket = str(ref.get("bucket") or "ph-supplemental")
    title_guess = str(ref.get("title") or _title_from_name(pdf_path))
    out_path = out_dir / f"{canonical_id}.json"
    if out_path.exists() and not overwrite:
        return out_path

    snippet = config.quick_text_snippet(str(pdf_path), max_chars=MAX_SNIPPET_CHARS)
    ref_labels = _load_reference_labels(ref_path)
    messages = _build_messages_with_labels(
        canonical_id=canonical_id,
        bucket_guess=bucket,
        title_guess=title_guess,
        text_snippet=snippet,
        candidate_labels=ref_labels,
    )
    raw = _chat_explainer(messages, model=model, temperature=temperature)
    payload = _coerce_payload(
        raw,
        canonical_id=canonical_id,
        bucket=bucket,
        title_guess=title_guess,
        source_pdf=pdf_path,
    )
    _safe_write(out_path, payload)
    return out_path


def _iter_pdfs(pdf_dir: Path, limit: int) -> List[Path]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if limit and limit > 0:
        return pdfs[:limit]
    return pdfs


def run_funsd(limit: int, overwrite: bool, model: Optional[str], temperature: Optional[float]) -> None:
    if not FUNSD_PDF_DIR.exists():
        print(f"[funsd] missing: {FUNSD_PDF_DIR}")
        return
    OUT_FUNSD.mkdir(parents=True, exist_ok=True)
    pdfs = _iter_pdfs(FUNSD_PDF_DIR, limit)
    print(f"[funsd] processing {len(pdfs)} PDF(s)")
    for idx, pdf in enumerate(pdfs, 1):
        out_path = _process_pdf(
            pdf,
            dataset="funsd-test",
            out_dir=OUT_FUNSD,
            ref_dir=REF_FUNSD,
            bucket="funsd-test",
            overwrite=overwrite,
            model=model,
            temperature=temperature,
        )
        print(f"  [{idx}/{len(pdfs)}] {pdf.name} -> {out_path.name}")


def run_ph(limit: int, overwrite: bool, model: Optional[str], temperature: Optional[float]) -> None:
    if not PH_PDF_DIR.exists():
        print(f"[ph] missing: {PH_PDF_DIR}")
        return
    OUT_PH.mkdir(parents=True, exist_ok=True)
    # Prefer reference-driven generation to avoid hash drift
    ref_files = sorted(REF_PH.glob("*.json"))
    if limit and limit > 0:
        ref_files = ref_files[:limit]
    print(f"[ph] processing {len(ref_files)} reference(s)")
    for idx, refp in enumerate(ref_files, 1):
        out_path = _process_via_reference(
            refp,
            out_dir=OUT_PH,
            overwrite=overwrite,
            model=model,
            temperature=temperature,
        )
        name = refp.name
        print(f"  [{idx}/{len(ref_files)}] {name} -> {out_path.name if out_path else 'skip'}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate explainer JSONs for evaluation.")
    ap.add_argument("--dataset", choices=["all", "funsd", "ph"], default="all")
    ap.add_argument("--limit-funsd", type=int, default=0)
    ap.add_argument("--limit-ph", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="Re-generate even if output exists.")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dataset in ("all", "funsd"):
        run_funsd(args.limit_funsd, args.overwrite, args.model, args.temperature)
    if args.dataset in ("all", "ph"):
        run_ph(args.limit_ph, args.overwrite, args.model, args.temperature)


if __name__ == "__main__":
    main()
