#!/usr/bin/env python3
"""
Batch-generate gold reference explanation JSONs for:
  • FUNSD test PDFs (data/funsd/pdfs/test)
  • PH supplemental forms (data/ph_trained/ph-supplemental-forms)

Output lives under explanations/refs/<dataset>/ and follows an explainer-like schema
with per-field "reference" text optimized for ROUGE/METEOR evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import base64

from scripts import config

# --- paths ---
BASE = config.BASE_DIR
FUNSD_TEST_DIR = BASE / "data" / "funsd" / "pdfs" / "test"
PH_SUPP_DIR = BASE / "data" / "ph_trained" / "ph-supplemental-forms"

# Research dashboard reference locations (per request)
DASH_ROOT = BASE / "static" / "research_dashboard"
FUNSD_OUT_DIR = DASH_ROOT / "funsd" / "references"
PH_OUT_DIR = DASH_ROOT / "ph_trained" / "references"

MAX_SNIPPET_CHARS = int(os.getenv("INTELLIFORM_REF_SNIPPET_MAX_CHARS", "12000") or 12000)

# --- prompt pieces ---
REFERENCE_SCHEMA_NOTES = f"""
You are generating a gold-standard reference explanation JSON for a PDF form. Output MUST be valid JSON:
{{
  "title": <string>,
  "form_id": <string>,
  "canonical_id": <string>,
  "bucket": <string>,
  "schema_version": 1,
  "created_at": <ISO8601>,
  "updated_at": <ISO8601>,
  "sections": [
    {{
      "title": <string>,
      "fields": [
        {{"label": <string>, "reference": <string>}}
      ]
    }}
  ],
  "fields": [
    {{"label": <string>, "reference": <string>, "section": <string>}}
  ],
  "notes": "reference_explainer_v1"
}}

Follow this exact shape (example content is illustrative only; do NOT copy values):
{{
  "title": "Form Title",
  "form_id": "form-title",
  "canonical_id": "abc123",
  "bucket": "funsd-test",
  "schema_version": 1,
  "created_at": "2024-11-28T00:00:00Z",
  "updated_at": "2024-11-28T00:00:00Z",
  "sections": [
    {{
      "title": "Section Title",
      "fields": [
        {{"label": "Printed Label", "reference": "Instruction to user"}}
      ]
    }}
  ],
  "fields": [
    {{"label": "Printed Label", "reference": "Instruction to user", "section": "Section Title"}}
  ],
  "notes": "reference_explainer_v1"
}}

STRICT RULES (align with IntelliForm explainer schema, but this is a REFERENCE so omit metrics):
- Follow all label/section grouping norms from the explainer schema below.
- Use exact printed labels/headings; trim only trailing punctuation/colons. One printed label → one field.
- Reference text = concise, imperative guidance on what to write/check; include formats + explicit examples for dates/IDs/amounts.
- Maximize lexical overlap with printed wording so ROUGE/METEOR stay high, but never hallucinate unseen content.
- Include EVERY answer prompt plus any “For Official Use/Office Only” areas (set reference like "For office use only; leave blank").
- Order fields left→right then top→bottom; group by visible section/row headers.
- If a label is unreadable/missing, keep the label if visible and set reference to "N/A".
- No markdown or prose outside the JSON object.

Explainer schema guidance for consistency (do NOT include metrics in this reference JSON):
{config.EXPLAINER_SCHEMA_NOTES.strip()}
""".strip()

REFERENCE_SYSTEM_PROMPT = (
    "You are IntelliForm’s reference-writer. "
    "You craft gold, non-hallucinated reference instructions that mirror on-page wording to score well on ROUGE/METEOR. "
    "You must emit exactly one JSON object following the schema and rules. "
    "This JSON will be used as the REFERENCE against model-generated explainers, so stay consistent with IntelliForm’s label/section conventions and omit metrics."
)


# --- helpers ---
def _iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_json_write(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _title_from_name(path: Path) -> str:
    s = path.stem
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(s.split()).strip()


def _bucket_from_ph_name(path: Path) -> str:
    stem = path.stem.lower()
    for key in ("government", "banking", "finance", "health", "healthcare", "insurance"):
        if key in stem:
            return key
    return "ph-supplemental"


def _page_image_b64(pdf_path: Path, page_index: int = 0, scale: float = 2.0) -> Optional[str]:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        if page_index >= len(doc):
            return None
        page = doc[page_index]
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        data = pix.tobytes("png")
        doc.close()
        return base64.b64encode(data).decode("ascii")
    except Exception:
        return None


def _build_reference_messages(
    *,
    canonical_id: str,
    bucket: str,
    title_guess: str,
    dataset: str,
    text_snippet: str,
    candidate_labels: Optional[List[str]] = None,
    image_b64: Optional[str] = None,
) -> List[Dict[str, Any]]:
    labels_hint = ""
    if candidate_labels:
        uniq = [x for x in sorted({(x or "").strip() for x in candidate_labels}) if x]
        if uniq:
            labels_hint = "Candidate labels (from detector/annos):\n- " + "\n- ".join(uniq[:120]) + "\n"

    base_text = f"""
Generate the gold reference explanation JSON for evaluation.

Canonical template hash (ID): {canonical_id}
Dataset: {dataset}
Bucket: {bucket}
Title guess: {title_guess}

Primary evidence: OCR/visible text excerpt (may be empty or partial):
---
{text_snippet}
---

Use ONLY what you can verify from the provided text and the attached page image (if present). Keep phrasing close to the document so lexical metrics stay strong.

{labels_hint}
Schema & strict rules:
{REFERENCE_SCHEMA_NOTES}

Output ONLY the JSON.
""".strip()

    content: List[Any] = [{"type": "text", "text": base_text}]
    if image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})

    return [
        {"role": "system", "content": REFERENCE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _coerce_reference_payload(
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

    now = _iso_utc()
    data.setdefault("title", title_guess or canonical_id)
    data.setdefault("form_id", config.sanitize_form_id(title_guess or canonical_id))
    data["canonical_id"] = canonical_id
    data["bucket"] = bucket
    data.setdefault("schema_version", 1)
    data.setdefault("created_at", now)
    data["updated_at"] = now
    data.setdefault("notes", "reference_explainer_v1")
    data["source_pdf"] = str(source_pdf.relative_to(BASE))

    # Normalize sections and flatten to fields for easy scoring.
    sections = data.get("sections") or []
    if not isinstance(sections, list):
        sections = []
    flat_fields: List[Dict[str, Any]] = []
    for sec in sections:
        sec_title = ""
        if isinstance(sec, dict):
            sec_title = str(sec.get("title", "") or "").strip()
            for f in sec.get("fields") or []:
                if not isinstance(f, dict):
                    continue
                label = str(f.get("label", "") or "").strip()
                ref = str(f.get("reference", "") or f.get("summary", "") or "").strip()
                if not label:
                    continue
                flat_fields.append(
                    {"label": label, "reference": ref or "N/A", "section": sec_title}
                )
    if flat_fields:
        data["fields"] = flat_fields
    else:
        data.setdefault("fields", [])
    return data


def _process_pdf(
    pdf_path: Path,
    *,
    dataset: str,
    bucket: str,
    out_dir: Path,
    overwrite: bool = False,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Path:
    canonical_id = config.canonical_template_hash(pdf_path)
    out_path = out_dir / f"{canonical_id}.json"
    if out_path.exists() and not overwrite:
        return out_path

    title_guess = _title_from_name(pdf_path)
    snippet = config.quick_text_snippet(str(pdf_path), max_chars=MAX_SNIPPET_CHARS)
    # include first-page image when OCR is weak
    img_b64 = _page_image_b64(pdf_path, page_index=0, scale=2.0)
    messages = _build_reference_messages(
        canonical_id=canonical_id,
        bucket=bucket,
        title_guess=title_guess,
        dataset=dataset,
        text_snippet=snippet,
        candidate_labels=None,
        image_b64=img_b64,
    )

    raw = config.chat_completion(
        model=model or config.ENGINE_MODEL,
        messages=messages,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE if temperature is None else float(temperature),
        enforce_json=True,
    )
    payload = _coerce_reference_payload(
        raw,
        canonical_id=canonical_id,
        bucket=bucket,
        title_guess=title_guess,
        source_pdf=pdf_path,
    )
    _safe_json_write(out_path, payload)
    return out_path


def _iter_pdfs(pdf_dir: Path, limit: int) -> List[Path]:
    pdfs = sorted([p for p in pdf_dir.glob("*.pdf")])
    if limit and limit > 0:
        return pdfs[:limit]
    return pdfs


def run_funsd(limit: int, overwrite: bool, model: Optional[str], temperature: Optional[float]) -> None:
    if not FUNSD_TEST_DIR.exists():
        print(f"[funsd] missing folder: {FUNSD_TEST_DIR}")
        return
    FUNSD_OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = _iter_pdfs(FUNSD_TEST_DIR, limit)
    if not pdfs:
        print("[funsd] no PDFs found.")
        return
    print(f"[funsd] processing {len(pdfs)} PDF(s)")
    for idx, pdf in enumerate(pdfs, 1):
        out_path = _process_pdf(
            pdf,
            dataset="funsd-test",
            bucket="funsd-test",
            out_dir=FUNSD_OUT_DIR,
            overwrite=overwrite,
            model=model,
            temperature=temperature,
        )
        print(f"  [{idx}/{len(pdfs)}] {pdf.name} -> {out_path.name}")


def run_ph_supp(limit: int, overwrite: bool, model: Optional[str], temperature: Optional[float]) -> None:
    if not PH_SUPP_DIR.exists():
        print(f"[ph] missing folder: {PH_SUPP_DIR}")
        return
    PH_OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = _iter_pdfs(PH_SUPP_DIR, limit)
    if not pdfs:
        print("[ph] no PDFs found.")
        return
    print(f"[ph] processing {len(pdfs)} PDF(s)")
    for idx, pdf in enumerate(pdfs, 1):
        bucket = _bucket_from_ph_name(pdf)
        out_path = _process_pdf(
            pdf,
            dataset="ph-supplemental",
            bucket=bucket,
            out_dir=PH_OUT_DIR,
            overwrite=overwrite,
            model=model,
            temperature=temperature,
        )
        print(f"  [{idx}/{len(pdfs)}] {pdf.name} -> {out_path.name}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate reference explanation JSONs for FUNSD test + PH supplemental PDFs via GPT."
    )
    ap.add_argument("--limit-funsd", type=int, default=0, help="Limit FUNSD test items (0 = all).")
    ap.add_argument("--limit-ph", type=int, default=0, help="Limit PH supplemental items (0 = all).")
    ap.add_argument("--overwrite", action="store_true", help="Re-generate even if output JSON exists.")
    ap.add_argument("--model", type=str, default=None, help="Override model (defaults to config.ENGINE_MODEL).")
    ap.add_argument("--temperature", type=float, default=None, help="Temperature override.")
    ap.add_argument(
        "--dataset",
        choices=["all", "funsd", "ph"],
        default="all",
        help="Choose which dataset(s) to process.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dataset in ("all", "funsd"):
        run_funsd(args.limit_funsd, args.overwrite, args.model, args.temperature)
    if args.dataset in ("all", "ph"):
        run_ph_supp(args.limit_ph, args.overwrite, args.model, args.temperature)


if __name__ == "__main__":
    main()
