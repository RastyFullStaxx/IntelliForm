# services/overlay_renderer.py
# Renders per-page PNG overlays from annotations JSON using PyMuPDF (fitz).

from __future__ import annotations
import os, json
from typing import List, Dict, Any, Iterable, Tuple
import fitz  # PyMuPDF

def _normalize_boxes(items: Iterable[Dict[str, Any]]) -> Dict[int, List[Tuple[float, float, float, float]]]:
    by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for it in items:
        bbox = it.get("bbox")
        if not bbox or len(bbox) != 4: continue
        p = int(it.get("page", 0))
        x0, y0, x1, y1 = [float(v) for v in bbox]
        if x1 <= x0 or y1 <= y0: continue
        by_page.setdefault(p, []).append((x0, y0, x1, y1))
    return by_page

def render_overlays(
    pdf_path: str,
    ann_path: str,
    out_dir: str,
    *,
    dpi: int = 180,
    stroke_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    stroke_width: float = 1.2
) -> List[str]:
    if not (os.path.exists(pdf_path) and os.path.exists(ann_path)):
        return []
    os.makedirs(out_dir, exist_ok=True)

    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f) or {}

    if isinstance(ann.get("groups"), list) and ann["groups"]:
        by_page = _normalize_boxes(ann["groups"])
    elif isinstance(ann.get("tokens"), list) and ann["tokens"]:
        by_page = _normalize_boxes(ann["tokens"])
    else:
        by_page = {}

    doc = fitz.open(pdf_path)
    out_paths: List[str] = []
    try:
        for pno in range(len(doc)):
            page = doc[pno]
            rects = by_page.get(pno, [])
            if rects:
                shape = page.new_shape()
                for (x0, y0, x1, y1) in rects:
                    shape.draw_rect(fitz.Rect(x0, y0, x1, y1))
                shape.finish(color=stroke_rgb, fill=None, width=stroke_width)
                shape.commit()
            out_png = os.path.join(out_dir, f"page-{pno + 1:02}.png")
            page.get_pixmap(dpi=dpi).save(out_png)
            out_paths.append(out_png)
    finally:
        doc.close()
    return out_paths
