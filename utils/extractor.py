# utils/extractor.py

"""
IntelliForm PDF Extractor
=========================

WHAT THIS MODULE DOES
---------------------
Handles layout-aware extraction of tokens and bounding boxes from PDF files.
Uses **pdfplumber** to parse text directly from page objects (no OCR by default).

Outputs are normalized to LayoutLM-style coordinates (0–1000 range), making
them directly usable by downstream models such as LayoutLMv3.

INTEGRATED MODULES
------------------
- pdfplumber : low-level text & geometry extraction
- dataclasses : structured containers for tokens, pages, results

RETURNED OBJECTS
----------------
- ExtractResult
    - pages          : list of PageInfo (width, height, page index)
    - tokens         : flat list of Token(text, bbox, page)
    - tokens_by_page : grouped tokens per page

EXAMPLE USAGE
-------------
from utils.extractor import extract_pdf

result = extract_pdf("uploads/sample.pdf")
print(result.tokens[0].text, result.tokens[0].bbox)

INTERACTIONS
------------
- Called by: utils.llmv3_infer (for runtime inference)
- Used indirectly by: scripts/train_classifier.py (if generating synthetic data)

SECURITY / DEPLOY NOTES
-----------------------
- Always validate input paths; restrict to `uploads/`.
- For large PDFs, consider batching pages to manage memory.
- OCR can be added if required, but default is off for speed/consistency.
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import List, Tuple
import pdfplumber
import os


@dataclass
class Token:
    text: str
    bbox: Tuple[int, int, int, int]  # normalized [x0,y0,x1,y1]
    page: int


@dataclass
class PageInfo:
    width: float
    height: float
    page_num: int


@dataclass
class ExtractResult:
    pages: List[PageInfo]
    tokens: List[Token]               # flat list of tokens
    tokens_by_page: List[List[Token]] # per-page grouping


def _normalize_bbox(
    x0: float, y0: float, x1: float, y1: float,
    width: float, height: float
) -> Tuple[int, int, int, int]:
    """
    Normalize absolute PDF coordinates into 0–1000 LayoutLM-style range.
    """
    def clamp(v, lo=0, hi=1000): return max(lo, min(hi, v))
    nx0 = int(round((x0 / width) * 1000))
    ny0 = int(round((y0 / height) * 1000))
    nx1 = int(round((x1 / width) * 1000))
    ny1 = int(round((y1 / height) * 1000))
    nx0, nx1 = min(nx0, nx1), max(nx0, nx1)
    ny0, ny1 = min(ny0, ny1), max(ny0, ny1)
    return (clamp(nx0), clamp(ny0), clamp(nx1), clamp(ny1))


def extract_pdf(
    pdf_path: str,
    dedupe_whitespace: bool = True,
    min_len: int = 1,
) -> ExtractResult:
    """
    Extract tokens and bounding boxes from a PDF using pdfplumber.

    Args:
        pdf_path (str): Path to the PDF file.
        dedupe_whitespace (bool): Collapse multiple spaces inside words.
        min_len (int): Minimum token length to keep.

    Returns:
        ExtractResult: structured output containing tokens, bboxes, and page info.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: List[PageInfo] = []
    tokens_flat: List[Token] = []
    tokens_per_page: List[List[Token]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for p_idx, page in enumerate(pdf.pages):
            width = float(page.width)
            height = float(page.height)
            pages.append(PageInfo(width=width, height=height, page_num=p_idx))

            words = page.extract_words(x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False)
            page_tokens: List[Token] = []

            for w in words or []:
                text = w.get("text", "")
                if dedupe_whitespace:
                    text = " ".join(text.split())
                if len(text) < min_len:
                    continue

                x0, x1 = float(w["x0"]), float(w["x1"])
                top, bottom = float(w["top"]), float(w["bottom"])
                bbox = _normalize_bbox(x0, top, x1, bottom, width, height)

                tok = Token(text=text, bbox=bbox, page=p_idx)
                page_tokens.append(tok)
                tokens_flat.append(tok)

            tokens_per_page.append(page_tokens)

    return ExtractResult(pages=pages, tokens=tokens_flat, tokens_by_page=tokens_per_page)
