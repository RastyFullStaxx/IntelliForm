# scripts/view_json_overlay.py
from __future__ import annotations
import os, re, json, argparse, hashlib, colorsys
from typing import Tuple, Optional, Dict, Iterable, Set, List
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

def parse_base_and_page(json_path: str) -> Tuple[str, int]:
    base = os.path.splitext(os.path.basename(json_path))[0]
    m = re.search(r"_p(\d{3})$", base)
    if not m:
        raise ValueError(f"Cannot parse page index from {base} (expect ..._pNNN)")
    page_idx = int(m.group(1)) - 1
    pdf_stem = base[: m.start()]  # remove _pNNN
    return pdf_stem, page_idx

def find_pdf(pdf_root: str, pdf_stem: str) -> Optional[str]:
    target = f"{pdf_stem}.pdf".lower()
    for r, _, files in os.walk(pdf_root):
        for f in files:
            if f.lower() == target:
                return os.path.join(r, f)
    return None

def denorm_bbox(box, w, h):
    x0 = int(round(box[0] / 1000.0 * w))
    y0 = int(round(box[1] / 1000.0 * h))
    x1 = int(round(box[2] / 1000.0 * w))
    y1 = int(round(box[3] / 1000.0 * h))
    return (x0, y0, x1, y1)

def entity_type(lbl: str) -> str:
    if lbl == "O": return "O"
    return lbl.split("-", 1)[-1] if "-" in lbl else lbl

def types_from_labels_map(path: Optional[str]) -> Set[str]:
    out: Set[str] = set()
    if not path or not os.path.exists(path): return out
    m = json.load(open(path, "r", encoding="utf-8"))
    for k in m.keys():
        if k == "O": continue
        out.add(entity_type(k))
    return out

def types_from_annotations(json_paths: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for jp in json_paths:
        try:
            ex = json.load(open(jp, "r", encoding="utf-8"))
        except Exception:
            continue
        for k in ex.get("labels", []):
            if k == "O": continue
            out.add(entity_type(k))
    return out

def stable_color(name: str) -> tuple[int,int,int]:
    """Deterministic pastel-like color from string."""
    h = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16)
    hue = (h % 360) / 360.0
    sat = 0.60
    val = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(r*255), int(g*255), int(b*255))

def build_palette(all_types: Set[str]) -> Dict[str, tuple[int,int,int]]:
    pal = {}
    for t in sorted(all_types):
        pal[t] = stable_color(t)
    return pal

def draw_legend(draw: ImageDraw.ImageDraw, palette: Dict[str, tuple], start_xy=(10,10), per_col=12, pad=6, swatch=14, font=None):
    types = sorted([t for t in palette.keys()])
    if not types: return
    x0, y0 = start_xy
    col = 0; row = 0
    for i, t in enumerate(types):
        if row >= per_col:
            col += 1; row = 0
        x = x0 + col*220
        y = y0 + row*(swatch + pad + 4)
        draw.rectangle([x, y, x+swatch, y+swatch], fill=palette[t], outline=(0,0,0))
        draw.text((x+swatch+8, y-2), t, fill=(0,0,0), font=font)
        row += 1

def draw_overlay(pdf_path: str, page_idx: int, ann: dict, out_path: str, palette: Dict[str, tuple], zoom: float = 2.0, show_text: bool=False):
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        w, h = pix.width, pix.height
        boxes = ann.get("bboxes") or ann.get("bbox") or []
        labels = ann.get("labels", ["O"] * len(boxes))

        for b, lab in zip(boxes, labels):
            et = entity_type(lab)
            if et == "O":
                col = (128,128,128)
            else:
                col = palette.get(et, stable_color(et))
            x0, y0, x1, y1 = denorm_bbox(b, w, h)
            draw.rectangle([x0, y0, x1, y1], outline=col, width=2)
            if show_text and et != "O":
                draw.text((x0+2, y0+2), et, fill=col, font=font)

        # legend box background
        legend_w = 460; legend_h = 260  # grows as needed; simple fixed bg
        draw.rectangle([6,6,6+legend_w,6+legend_h], fill=(255,255,255), outline=(200,200,200))
        draw_legend(draw, {k:v for k,v in palette.items()}, start_xy=(10,10), per_col=12, font=font)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        img.save(out_path, "PNG")
    finally:
        doc.close()

def collect_jsons(arg_json: Optional[str], arg_glob: Optional[str]) -> List[str]:
    if arg_json:
        return [arg_json]
    if arg_glob:
        import glob
        return glob.glob(arg_glob)
    raise SystemExit("Provide --json or --glob")

def main():
    ap = argparse.ArgumentParser(description="Render JSON annotation overlays with dynamic colors from labels_union.json.")
    ap.add_argument("--json", help="Single JSON to render.")
    ap.add_argument("--glob", help="Glob for multiple, e.g. data\\ph_forms_all\\annotations\\*.json")
    ap.add_argument("--pdf_root", required=True, help="Root folder containing PDFs.")
    ap.add_argument("--labels_map", default="data/labels_union.json", help="labels_union.json (optional but recommended).")
    ap.add_argument("--out_dir", default="out/overlays")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--show_text", action="store_true")
    args = ap.parse_args()

    # gather files
    json_paths = collect_jsons(args.json, args.glob)
    if args.limit:
        json_paths = sorted(json_paths)[: args.limit]

    # build dynamic palette
    types = types_from_labels_map(args.labels_map)
    types |= types_from_annotations(json_paths)
    if "O" in types:
        types.remove("O")
    palette = build_palette(types)

    # render each
    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            ann = json.load(f)

        # prefer meta.source_pdf if provided; else infer by stem
        pdf_path = None
        meta = ann.get("meta") or {}
        src_rel = meta.get("source_pdf")
        if src_rel:
            cand = os.path.join(args.pdf_root, src_rel)
            if os.path.exists(cand):
                pdf_path = cand

        if not pdf_path:
            stem, page_idx = parse_base_and_page(jp)
            pdf_path = find_pdf(args.pdf_root, stem)
            if pdf_path is None:
                print(f"[skip] PDF not found for {jp}")
                continue
        else:
            # if we used meta, still need page index
            _, page_idx = parse_base_and_page(jp)

        out_name = os.path.splitext(os.path.basename(jp))[0] + ".png"
        out_path = os.path.join(args.out_dir, out_name)
        draw_overlay(pdf_path, page_idx, ann, out_path, palette, zoom=args.zoom, show_text=args.show_text)
        print("[ok] wrote", out_path)

if __name__ == "__main__":
    main()
