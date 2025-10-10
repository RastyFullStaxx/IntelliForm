# api.py
"""
IntelliForm API Routes (FastAPI)
================================

WHAT THIS MODULE DOES
---------------------
- Serves the web-facing API for IntelliForm.
- Uses a TEMPLATE-BASED canonical hash (robust to filled values) as the true ID.
- Keeps the facade behavior:
    * Prelabeler runs for overlays on every upload/open
    * Known form → load curated explainer from explanations/<bucket>/<HASH>.json
    * Unknown form → generate via LLM fallback and save under the hash
- Registry (/panel) is keyed by HASH.

PRIMARY ENDPOINTS
-----------------
- GET  /api/health
- POST /api/upload          → saves PDF, computes template-hash, returns canonical_form_id
- POST /api/prelabel        → runs prelabeler, promotes <HASH>__temp.json → <HASH>.json, renders overlays
- POST /api/explainer       → (legacy form) accepts a PDF blob + bucket + form_id; generates explainer via fallback
- POST /api/explainer.ensure (new) ensures an explainer exists for {hash,bucket,...}; creates via fallback if missing
- GET  /panel               → returns explanations/registry.json
- GET  /api/metrics         → quick metrics text (unchanged)

SECURITY / DEPLOY NOTES
-----------------------
- Validate /uploads paths.
- Tighten CORS in prod.
"""

# api.py
from __future__ import annotations

import os
import uuid
import time
import json
import shutil
from typing import Optional, List, Dict, Any
from pathlib import Path
import re
import fitz  # PyMuPDF, already in your project via overlay renderer

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scripts import config
from utils.llmv3_infer import analyze_pdf, analyze_tokens    # optional analyzer
from utils.dual_head import generate_explainer               # LLM fallback explainer
from services.overlay_renderer import render_overlays

# ✅ New: centralized registry helpers
from services.registry import (
    load_registry as reg_load,
    upsert_registry as reg_upsert,
    find_by_hash as reg_find,
)

# -----------------------------
# Paths & mounts (ORDER MATTERS)
# -----------------------------
BASE_DIR     = config.BASE_DIR
UPLOADS_DIR  = config.UPLOADS_DIR
EXPL_DIR     = config.EXPL_DIR
ANN_DIR      = config.ANNO_DIR
REG_PATH     = config.REGISTRY_PATH

STATIC_DIR   = BASE_DIR / "static"
METRICS_TXT  = STATIC_DIR / "metrics_report.txt"
OUT_OVERLAYS = BASE_DIR / "out" / "overlays"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
OUT_OVERLAYS.mkdir(parents=True, exist_ok=True)

# -----------------------------
# App & CORS
# -----------------------------
app = FastAPI(title="IntelliForm API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for prod
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Public mounts
app.mount("/uploads",      StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/explanations", StaticFiles(directory=str(EXPL_DIR)),    name="explanations")
app.mount("/static",       StaticFiles(directory=str(STATIC_DIR)),  name="static")

# -----------------------------
# Models
# -----------------------------
class AnalyzeTokensBody(BaseModel):
    tokens: List[str]
    bboxes: List[List[int]]
    page_ids: Optional[List[int]] = None
    device: Optional[str] = None
    min_confidence: Optional[float] = 0.0
    no_graph: Optional[bool] = False
    classifier: Optional[str] = "saved_models/classifier.pt"
    t5: Optional[str] = "saved_models/saved_models/t5.pt"

class EnsureExplainerBody(BaseModel):
    canonical_form_id: str               # TEMPLATE HASH
    bucket: str                          # e.g., healthcare / banking / government / tax
    human_title: Optional[str] = None    # pretty title for UI
    pdf_disk_path: Optional[str] = None  # optional; used by fallback for better context
    aliases: Optional[List[str]] = None  # optional extra aliases to save in registry

# -----------------------------
# Small helpers
# -----------------------------
def _quick_report(payload: dict, path: os.PathLike) -> None:
    fields = payload.get("fields", [])
    counts: Dict[str, int] = {}
    scores: List[float] = []
    for f in fields:
        lb = f.get("label", "UNK")
        counts[lb] = counts.get(lb, 0) + 1
        try:
            if "score" in f:
                scores.append(float(f["score"]))
        except Exception:
            pass
    avg = (sum(scores) / len(scores)) if scores else 0.0
    rt = payload.get("runtime", {})
    lines = [
        "IntelliForm — Metrics (Quick Report)",
        "====================================",
        f"Field Count: {len(fields)}",
        "",
        "Counts by Label:",
    ]
    for k in sorted(counts.keys()):
        lines.append(f"  - {k}: {counts[k]}")
    lines += [
        "",
        f"Average Score: {avg:.3f}",
        "",
        "Runtimes (ms):",
        f"  - extract   : {rt.get('extract_ms','-')}",
        f"  - classify  : {rt.get('classify_ms','-')}",
        f"  - summarize : {rt.get('summarize_ms','-')}",
        "",
    ]
    (STATIC_DIR / os.fspath(path)).write_text("\n".join(lines), encoding="utf-8") if isinstance(path, str) else \
        path.write_text("\n".join(lines), encoding="utf-8")

def _uploads_web_to_disk(path: str) -> str:
    p = str(path).replace("\\", "/")
    if p.startswith("/uploads/"):
        tail = p[len("/uploads/"):]
        return str(UPLOADS_DIR / tail)
    if p.startswith("/static/uploads/"):  # legacy safety
        tail = p[len("/static/uploads/"):]
        return str(UPLOADS_DIR / tail)
    return path  # may already be a disk path

def _validate_upload_path(path: str) -> str:
    """
    Ensures the given path is inside <repo>/uploads and exists.
    Accepts either a web path (/uploads/...) or a full disk path under UPLOADS_DIR.
    """
    disk = os.path.normpath(_uploads_web_to_disk(path))
    root = os.path.normpath(str(UPLOADS_DIR))
    if not disk.startswith(root):
        raise HTTPException(status_code=400, detail="Path must be under /uploads.")
    if not os.path.exists(disk):
        raise HTTPException(status_code=404, detail="File not found.")
    return disk

def _extract_embedded_form_id(pdf_path: str) -> Optional[str]:
    """
    Read PDF metadata and extract IntelliForm-embedded form_id, if any.
    We set this in the client export as:
        Subject: "IntelliForm-FormId:<hash>"
    """
    try:
        with fitz.open(pdf_path) as doc:
            subj = (doc.metadata or {}).get("subject") or ""
    except Exception:
        return None
    m = re.search(r"IntelliForm-FormId:([A-Za-z0-9_-]{8,128})", subj)
    return m.group(1) if m else None

# -----------------------------
# Routes
# -----------------------------
@app.get("/api/health")
def api_health():
    return {"status": "ok", "mode": ("pipeline" if config.LIVE_MODE else "static"), "time": int(time.time())}

@app.get("/panel")
def panel():
    data = reg_load(str(REG_PATH))
    return JSONResponse(content=data)

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """
    Save PDF → compute TEMPLATE hash → return canonical_form_id.
    If the PDF carries our embedded Subject "IntelliForm-FormId:<hash>",
    prefer that as the canonical_form_id so edited PDFs map to the same explainer.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = file.filename.replace(" ", "_")
    uid = uuid.uuid4().hex
    stored = f"{uid}_{safe_name}"  # file_id used by the frontend

    disk_path = str(UPLOADS_DIR / stored)
    web_path  = f"/uploads/{stored}"

    try:
        with open(disk_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    # 1) Compute TEMPLATE-based hash from file content (robust to filled values)
    template_hash = config.canonical_template_hash(disk_path)

    # 2) Prefer embedded IntelliForm FormId (for edited PDFs we exported client-side)
    embedded_form_id = _extract_embedded_form_id(disk_path)
    canonical_form_id = embedded_form_id or template_hash

    # Return BOTH new and legacy fields for full compatibility
    return {
        "ok": True,
        "web_path": web_path,
        "disk_path": disk_path,
        # NEW truth (may come from embedded metadata if present):
        "canonical_form_id": canonical_form_id,
        # Legacy/compat (front-end still reads `form_id`):
        "form_id": canonical_form_id,
        "file_id": stored,
        "path": web_path,
    }

@app.post("/api/prelabel")
async def api_prelabel(
    form_id: str = Form(...),         # should be the TEMPLATE HASH from /api/upload
    pdf_disk_path: str = Form(...),
):
    """
    Runs prelabeler and promotes temp → canonical <HASH>.json.
    Renders overlays for the current PDF using the canonical annotation.
    """
    # Resolve PDF path
    try:
        pdf_disk_path = _uploads_web_to_disk(pdf_disk_path)
        disk = _validate_upload_path(pdf_disk_path)
    except HTTPException:
        disk = pdf_disk_path if os.path.exists(pdf_disk_path) else None
    if not disk:
        raise HTTPException(status_code=400, detail="Invalid PDF path.")

    # Ensure form_id *is* the TEMPLATE HASH (recompute if caller sent something else)
    hash_checked = config.canonical_template_hash(disk)
    if hash_checked != form_id:
        form_id = hash_checked  # enforce truth

    # Produce temp annotation JSON at <HASH>__temp.json
    temp_json = config.temp_annotation_path(form_id)
    try:
        launch = config.launch_prelabeler(pdf_path=disk, out_temp=temp_json)
    except Exception:
        launch = None  # still check file presence

    if not temp_json.exists():
        detail = getattr(launch, "stderr", None) if launch and not isinstance(launch, bool) else None
        raise HTTPException(status_code=500, detail=detail or "Prelabeler failed to produce temp annotation.")

    # Promote to canonical <HASH>.json
    promo = config.promote_or_reuse_annotation(form_id=form_id, temp_path=temp_json)
    if not promo.success or not promo.canonical_path:
        raise HTTPException(status_code=500, detail=promo.error or "Promotion failed.")

    canonical_path = promo.canonical_path
    canonical_form_id = form_id

    # Render overlays (grouped under the canonical hash for cache hygiene)
    overlays_dir = OUT_OVERLAYS / canonical_form_id
    try:
        overlays = render_overlays(
            pdf_path=disk,
            ann_path=str(canonical_path),
            out_dir=str(overlays_dir)
        )
    except Exception:
        overlays = []

    ann_web = f"/explanations/_annotations/{canonical_form_id}.json"

    return {
        "ok": True,
        "annotations": ann_web,
        "canonical_form_id": canonical_form_id,
        "overlays_dir": str(overlays_dir).replace("\\", "/"),
        "overlays": [p.replace("\\", "/") for p in overlays],
        "mode": ("pipeline" if config.LIVE_MODE else "static"),
    }

@app.post("/api/explainer")
async def api_explainer_legacy(
    file: UploadFile = File(...),
    bucket: str = Form(...),
    form_id: str = Form(...),           # SHOULD be the TEMPLATE HASH
    human_title: str = Form(...),
):
    """
    Legacy-compatible: accept a PDF blob, save it, then generate an explainer.
    **form_id must be the TEMPLATE HASH** (frontend should pass the value from /api/upload).
    """
    safe_name = file.filename.replace(" ", "_") or f"{form_id}.pdf"
    uid = uuid.uuid4().hex
    stored = f"{uid}_{safe_name}"
    pdf_disk_path = str(UPLOADS_DIR / stored)
    try:
        with open(pdf_disk_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    # Ensure form_id is indeed the template-hash of the file we just got
    hash_checked = config.canonical_template_hash(pdf_disk_path)
    if hash_checked != form_id:
        form_id = hash_checked

    out_dir = EXPL_DIR / bucket
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use LLM fallback to generate explainer (or refine existing)
    expl_path = generate_explainer(
        pdf_path=pdf_disk_path,
        bucket=bucket,
        form_id=form_id,           # HASH
        human_title=human_title,
        out_dir=str(out_dir),
    )

    # Store relative path for the client
    rel_path = os.path.relpath(expl_path, BASE_DIR).replace("\\", "/")
    reg_upsert(str(REG_PATH), form_id=form_id, title=human_title, rel_path=rel_path, bucket=bucket)

    return {
        "ok": True,
        "path": rel_path,
        "form_id": form_id,
        "title": human_title,
        "mode": ("pipeline" if config.LIVE_MODE else "static"),
    }

@app.post("/api/explainer.ensure")
async def api_explainer_ensure(body: EnsureExplainerBody):
    """
    Ensure an explainer exists for {canonical_form_id (HASH), bucket}.
    If missing → generate via LLM fallback and upsert the registry.
    Additionally: normalize legacy absolute paths in the registry to relative paths.
    """
    h = body.canonical_form_id
    bucket = body.bucket
    title = body.human_title or h
    pdf_disk_path = body.pdf_disk_path

    def _relativize(p: str) -> str:
        if not p:
            return p
        # If it's absolute (e.g., C:\... or /home/...), convert to BASE_DIR-relative.
        is_abs = os.path.isabs(p) or (":" in p.split("/")[0])  # crude Windows drive check when already slashified
        if is_abs:
            try:
                rel = os.path.relpath(p, BASE_DIR)
            except Exception:
                rel = p
            p = rel
        return p.replace("\\", "/").lstrip("/")  # serve under /<relpath>

    # If already present in registry, normalize its path if needed and return.
    existing = reg_find(str(REG_PATH), h)
    if existing:
        rel_path = _relativize(str(existing.get("path", "")))
        # If path changed (was absolute), upsert the normalized one.
        if rel_path and rel_path != existing.get("path"):
            reg_upsert(
                str(REG_PATH),
                form_id=h,
                title=(existing.get("title") or title),
                rel_path=rel_path,
                bucket=(existing.get("bucket") or bucket),
                aliases=(existing.get("aliases") or (body.aliases or [])),
            )
            existing = reg_find(str(REG_PATH), h) or {"form_id": h, "title": title, "path": rel_path, "bucket": bucket}
        return {
            "ok": True,
            "form_id": h,
            "title": existing.get("title", title),
            "path": existing.get("path"),
            "bucket": existing.get("bucket", bucket),
            "already_exists": True,
        }

    # Not in registry → create via fallback
    out_dir = EXPL_DIR / bucket
    out_dir.mkdir(parents=True, exist_ok=True)

    expl_path = generate_explainer(
        pdf_path=pdf_disk_path or "",
        bucket=bucket,
        form_id=h,         # HASH
        human_title=title,
        out_dir=str(out_dir),
    )
    rel_path = os.path.relpath(expl_path, BASE_DIR).replace("\\", "/")

    reg_upsert(
        str(REG_PATH),
        form_id=h,
        title=title,
        rel_path=rel_path,
        bucket=bucket,
        aliases=(body.aliases or []),
    )

    return {
        "ok": True,
        "form_id": h,
        "title": title,
        "path": rel_path,
        "bucket": bucket,
        "already_exists": False,
    }

@app.post("/api/analyze")
async def api_analyze(
    file_path: Optional[str] = Query(default=None, description="Web path (/uploads/...) or absolute path in uploads/"),
    body: Optional[AnalyzeTokensBody] = Body(default=None),
):
    """
    Optional analyzer for tokens/pdf (not required by the current UI flow).
    """
    try:
        t0 = time.time()
        if file_path:
            norm = _validate_upload_path(file_path)
            cfg = {
                "device": (body.device if body else ("cuda" if os.environ.get("USE_CUDA") == "1" else "cpu")),
                "min_confidence": (body.min_confidence if body else 0.0),
                "max_length": 512,
                "graph": {"use": not bool(body.no_graph if body else False), "strategy": "knn", "k": 8, "radius": None},
                "model_paths": {"classifier": (body.classifier if body else "saved_models/classifier.pt"),
                                "t5": (body.t5 if body else "saved_models/t5.pt")},
            }
            result = analyze_pdf(norm, config=cfg)
            title = os.path.basename(norm)
        else:
            if body is None:
                raise HTTPException(status_code=400, detail="Provide file_path or JSON body with tokens/bboxes.")
            cfg = {
                "device": body.device or ("cuda" if os.environ.get("USE_CUDA") == "1" else "cpu"),
                "min_confidence": body.min_confidence or 0.0,
                "max_length": 512,
                "graph": {"use": not bool(body.no_graph), "strategy": "knn", "k": 8, "radius": None},
                "model_paths": {"classifier": body.classifier or "saved_models/classifier.pt",
                                "t5": body.t5 or "saved_models/t5.pt"},
            }
            result = analyze_tokens(body.tokens, body.bboxes, page_ids=body.page_ids, config=cfg)
            title = "tokens.json"

        elapsed = time.time() - t0

        fields  = result.get("fields", [])
        pages   = result.get("pages") or result.get("num_pages")
        runtime = result.get("runtime", {})

        _quick_report(result, METRICS_TXT)

        return JSONResponse(content={
            "document": file_path or title,
            "title": title,
            "fields": fields,
            "metrics": {
                "precision": None, "recall": None, "f1": None,
                "rougeL": None, "meteor": None,
                "fields_count": len(fields),
                "processing_sec": elapsed,
                "pages": pages,
            },
            "runtime": runtime,
        })
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/metrics", response_class=PlainTextResponse)
def api_metrics():
    if not METRICS_TXT.exists():
        return PlainTextResponse("No metrics report found.", status_code=404)
    return PlainTextResponse(METRICS_TXT.read_text(encoding="utf-8"))
