# services/prelabeler.py
from __future__ import annotations
import os, json, shutil
from typing import Tuple
from scripts import config  # LIVE_MODE and launch_prelabeler

# NEW: import renderers
from services.overlay_renderer import render_overlays, render_gnn_visuals

def _stub_annotations(out_path: str) -> bool:
    stub = {
        "tokens": [{"text": "Sample", "bbox": [20, 20, 120, 50], "page": 0}],
        "groups": []
    }
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stub, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def _postprocess(pdf_path: str, ann_path: str, form_id: str) -> None:
    """Create out/overlay, out/gnn, out/prelabeled artifacts."""
    base_out = config.BASE_DIR / "out"
    overlay_dir = base_out / "overlay" / form_id
    gnn_dir     = base_out / "gnn" / form_id
    prel_dir    = base_out / "prelabeled"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    gnn_dir.mkdir(parents=True, exist_ok=True)
    prel_dir.mkdir(parents=True, exist_ok=True)

    # overlays
    try:
        render_overlays(str(pdf_path), str(ann_path), str(overlay_dir), dpi=180)
    except Exception:
        pass

    # gnn visuals
    try:
        render_gnn_visuals(str(pdf_path), str(ann_path), str(gnn_dir), strategy="knn", k=8, dpi=180)
    except Exception:
        pass

    # copy/save the prelabel json
    try:
        shutil.copyfile(ann_path, prel_dir / f"{form_id}.json")
    except Exception:
        pass

def ensure_annotations(pdf_path: str, form_id: str, ann_dir: str, base_dir: str) -> Tuple[bool, str]:
    os.makedirs(ann_dir, exist_ok=True)
    out_path = os.path.join(ann_dir, f"{form_id}.json")

    if os.path.exists(out_path):
        try: _postprocess(pdf_path, out_path, form_id)
        except Exception: pass
        return True, out_path

    if not config.LIVE_MODE:
        ok = _stub_annotations(out_path)
        if ok:
            try: _postprocess(pdf_path, out_path, form_id)
            except Exception: pass
        return (ok, out_path) if ok else (False, "")

    # Prefer in-process prelabel
    try:
        from utils.llmv3_infer import prelabel as _prelabel
        ok = _prelabel(pdf_path=str(pdf_path), out_json=str(out_path), dev=bool(config.DEV_MODE))
        if ok:
            try: _postprocess(pdf_path, out_path, form_id)
            except Exception: pass
            return True, out_path
    except Exception:
        pass

    # Fallback launcher
    try:
        ok = bool(config.launch_prelabeler(pdf_path=pdf_path, out_path=out_path, repo_root=base_dir))
        if ok:
            try: _postprocess(pdf_path, out_path, form_id)
            except Exception: pass
            return True, out_path
    except Exception:
        pass

    ok = _stub_annotations(out_path)
    if ok:
        try: _postprocess(pdf_path, out_path, form_id)
        except Exception: pass
    return (ok, out_path) if ok else (False, "")
