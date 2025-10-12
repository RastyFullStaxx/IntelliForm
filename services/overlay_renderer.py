# services/overlay_renderer.py
# Renders per-page PNG overlays from annotations JSON using PyMuPDF (fitz).

from __future__ import annotations
import os, json
from typing import List, Dict, Any, Iterable, Tuple, Optional
import fitz  # PyMuPDF

def _collect_boxes(items: Iterable[Dict[str, Any]]) -> Dict[int, List[Tuple[float,float,float,float]]]:
    """Return per-page list of normalized boxes [0..1000]."""
    by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for it in items or []:
        bbox = it.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        p = int(it.get("page", 0))
        x0, y0, x1, y1 = [float(v) for v in bbox]
        if x1 <= x0 or y1 <= y0:
            continue
        by_page.setdefault(p, []).append((x0, y0, x1, y1))
    return by_page

def _denorm_rect(b: Tuple[float,float,float,float], w: float, h: float) -> fitz.Rect:
    """Convert normalized [0..1000] box to page-space rect."""
    x0, y0, x1, y1 = b
    sx, sy = w / 1000.0, h / 1000.0
    return fitz.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)

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
        by_page_norm = _collect_boxes(ann["groups"])
    elif isinstance(ann.get("tokens"), list) and ann["tokens"]:
        by_page_norm = _collect_boxes(ann["tokens"])
    else:
        by_page_norm = {}

    doc = fitz.open(pdf_path)
    out_paths: List[str] = []
    try:
        for pno in range(len(doc)):
            page = doc[pno]
            w, h = float(page.rect.width), float(page.rect.height)
            rects = by_page_norm.get(pno, [])
            if rects:
                shape = page.new_shape()
                for nb in rects:
                    shape.draw_rect(_denorm_rect(nb, w, h))
                shape.finish(color=stroke_rgb, fill=None, width=stroke_width)
                shape.commit()
            out_png = os.path.join(out_dir, f"page-{pno + 1:02}.png")
            page.get_pixmap(dpi=dpi).save(out_png)
            out_paths.append(out_png)
    finally:
        doc.close()
    return out_paths

# ---- GNN visuals ----
try:
    from utils.graph_builder import build_edges
except Exception:
    build_edges = None

def render_gnn_visuals(
    pdf_path: str,
    ann_path: str,
    out_dir: str,
    *,
    strategy: str = "knn",
    k: int = 8,
    radius: Optional[float] = None,
    dpi: int = 180,
    line_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    line_width: float = 0.6,
) -> List[str]:
    if build_edges is None:
        return []
    if not (os.path.exists(pdf_path) and os.path.exists(ann_path)):
        return []
    os.makedirs(out_dir, exist_ok=True)

    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f) or {}
    tokens = ann.get("tokens") or []
    if not tokens:
        return []

    # group token centers (normalized) by page
    per_page: Dict[int, Tuple[list, list]] = {}
    for t in tokens:
        bb = t.get("bbox")
        p = int(t.get("page", 0))
        if not bb or len(bb) != 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in bb]
        if x1 <= x0 or y1 <= y0:
            continue
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        ent = per_page.setdefault(p, ([], []))
        ent[0].append((cx, cy))         # centers (normalized)
        ent[1].append([x0, y0, x1, y1]) # boxes   (normalized)

    import numpy as _np
    doc = fitz.open(pdf_path)
    out_paths: List[str] = []
    try:
        for pno in range(len(doc)):
            centers, bboxes = per_page.get(pno, ([], []))
            page = doc[pno]
            w, h = float(page.rect.width), float(page.rect.height)
            if centers and bboxes:
                Cn = _np.array(centers, dtype="float32")   # normalized centers
                Bn = _np.array(bboxes,  dtype="float32")   # normalized bboxes

                # Build edges in normalized space (what graph_builder expects)
                try:
                    g = build_edges(Bn, strategy=strategy, k=k, radius=radius, page_ids=None)
                    E = g["edge_index"].cpu().numpy().T
                except Exception:
                    E = _np.zeros((0, 2), dtype=int)

                # Denorm helper for centers
                sx, sy = w / 1000.0, h / 1000.0

                shape = page.new_shape()
                for i, j in E:
                    x0, y0 = Cn[i][0] * sx, Cn[i][1] * sy
                    x1, y1 = Cn[j][0] * sx, Cn[j][1] * sy
                    shape.draw_line(fitz.Point(float(x0), float(y0)), fitz.Point(float(x1), float(y1)))

                # also draw tiny center dots
                for c in Cn:
                    shape.draw_circle(fitz.Point(float(c[0] * sx), float(c[1] * sy)), 2.0)

                shape.finish(color=line_rgb, width=line_width)
                shape.commit()

            out_png = os.path.join(out_dir, f"page-{pno + 1:02}.png")
            page.get_pixmap(dpi=dpi).save(out_png)
            out_paths.append(out_png)
    finally:
        doc.close()
    return out_paths
