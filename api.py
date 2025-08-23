# api.py
"""
IntelliForm API Routes (FastAPI)
================================

WHAT THIS MODULE DOES
---------------------
Defines FastAPI endpoints for running IntelliForm inference and retrieving
results/metrics. This is the web-facing bridge to the backend pipeline.

INTEGRATED MODULES
------------------
- utils.extractor        : PDF token & layout extraction (if starting from raw PDF)
- utils.llmv3_infer      : Unified inference (classifier + T5 summarizer)
- utils.field_classifier : LayoutLMv3 + GNN + classifier (loaded inside llmv3_infer)
- utils.t5_summarize     : T5 summary generation (loaded inside llmv3_infer)
- utils.metrics          : Metric computation & report formatting (optional at inference)

PRIMARY ENDPOINTS
-----------------
- GET  /api/health
- POST /api/upload
- POST /api/analyze          (PDF path or tokens+bboxes)
- GET  /api/metrics          (quick text report)
- GET  /api/serve-file       (serve artifacts, e.g., metrics report)

RESPONSE SHAPE (ANALYZE)
------------------------
{
  "document": "uploads/2025-08-08_form.pdf",
  "title": "2025-08-08_form.pdf",
  "fields": [
    {"label":"FULL_NAME","score":0.93,"bbox":[x0,y0,x1,y1],"summary":"...","page":1,"group":"..."}
  ],
  "metrics": {
    "precision": null,
    "recall": null,
    "f1": 0.82,                    # if GT present (span IoU)
    "rougeL": 0.44,                # if GT present
    "meteor": 0.29,                # if GT present
    "fields_count": 27,            # always useful
    "processing_sec": 1.42,        # always useful
    "pages": 3                     # if known
  },
  "runtime": {"extract_ms":120,"classify_ms":85,"summarize_ms":40}
}

SECURITY / DEPLOY NOTES
-----------------------
- Validate file paths; restrict to `uploads/`.
- Enforce file size/type limits.
- Consider async or background tasks for large PDFs.
"""

from __future__ import annotations

import os
import uuid
import time
import shutil
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from pydantic import BaseModel

from utils.llmv3_infer import analyze_pdf, analyze_tokens
from utils.metrics import (
    evaluate_spans_iou,
    compute_rouge_meteor,
    format_metrics_for_report,
    save_report_txt,
)

# -----------------------------------------------------------------------------
# App & dirs
# -----------------------------------------------------------------------------
app = FastAPI(title="IntelliForm API", version="0.1.0")

# CORS: adjust origins for your frontend in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

UPLOAD_DIR = "uploads"
STATIC_DIR = "static"
METRICS_PATH = os.path.join(STATIC_DIR, "metrics_report.txt")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class AnalyzeTokensBody(BaseModel):
    tokens: List[str]
    bboxes: List[List[int]]
    page_ids: Optional[List[int]] = None
    # optional knobs
    device: Optional[str] = None
    min_confidence: Optional[float] = 0.0
    no_graph: Optional[bool] = False
    classifier: Optional[str] = "saved_models/classifier.pt"
    t5: Optional[str] = "saved_models/t5.pt"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _write_quick_report(result: dict, path: str = METRICS_PATH) -> None:
    fields = result.get("fields", [])
    counts: Dict[str, int] = {}
    scores: List[float] = []
    for f in fields:
        lbl = f.get("label", "UNK")
        counts[lbl] = counts.get(lbl, 0) + 1
        if "score" in f:
            try:
                scores.append(float(f["score"]))
            except Exception:
                pass
    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    rt = result.get("runtime", {})
    lines = []
    lines.append("IntelliForm — Metrics (Quick Report)")
    lines.append("====================================")
    lines.append(f"Document   : {result.get('document','')}")
    lines.append(f"Field Count: {len(fields)}")
    lines.append("")
    lines.append("Counts by Label:")
    for k in sorted(counts.keys()):
        lines.append(f"  - {k}: {counts[k]}")
    lines.append("")
    lines.append(f"Average Score: {avg_score:.3f}")
    lines.append("")
    lines.append("Runtimes (ms):")
    lines.append(f"  - extract   : {rt.get('extract_ms','-')}")
    lines.append(f"  - classify  : {rt.get('classify_ms','-')}")
    lines.append(f"  - summarize : {rt.get('summarize_ms','-')}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def _cfg_from_knobs(
    device: Optional[str],
    min_conf: Optional[float],
    no_graph: Optional[bool],
    classifier: Optional[str],
    t5: Optional[str],
) -> dict:
    return {
        "device": device or ("cuda" if os.environ.get("USE_CUDA") == "1" else "cpu"),
        "min_confidence": (min_conf if min_conf is not None else 0.0),
        "max_length": 512,
        "graph": {"use": not bool(no_graph), "strategy": "knn", "k": 8, "radius": None},
        "model_paths": {
            "classifier": classifier or "saved_models/classifier.pt",
            "t5": t5 or "saved_models/t5.pt",
        },
    }

def _validate_under_uploads(path: str) -> str:
    norm = os.path.normpath(path)
    if not norm.startswith(os.path.normpath(UPLOAD_DIR)):
        raise HTTPException(status_code=400, detail="Path must be under uploads/.")
    if not os.path.exists(norm):
        raise HTTPException(status_code=404, detail="File not found.")
    return norm

def _load_gt_for(pdf_path: str) -> List[Dict[str, Any]]:
    """
    OPTIONAL: implement your own GT fetch here.
    For example: look for a JSON next to the PDF: uploads/.../file.gt.json
    Should return: [{"label": "...", "bbox": [x0,y0,x1,y1], "reference_summary": "..."}]
    Return [] if none.
    """
    # Stub: no ground truth by default
    return []

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "time": int(time.time())}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    fid = f"{uuid.uuid4().hex}_{file.filename.replace(' ', '_')}"
    out_path = os.path.join(UPLOAD_DIR, fid)
    try:
        with open(out_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()
    # Return a path that the frontend can GET directly
    return {"file_id": fid, "path": out_path}

@app.post("/api/analyze")
async def analyze(
    file_path: Optional[str] = Query(default=None, description="Path under uploads/ to a PDF"),
    body: Optional[AnalyzeTokensBody] = Body(default=None),
):
    """
    Two modes:
      - PDF:   POST /api/analyze?file_path=uploads/....pdf
      - Tokens: POST JSON body with tokens+bboxes (page_ids optional)
    """
    try:
        t0 = time.time()
        if file_path:
            norm = _validate_under_uploads(file_path)
            cfg = _cfg_from_knobs(
                device=(body.device if body else None),
                min_conf=(body.min_confidence if body else None),
                no_graph=(body.no_graph if body else None),
                classifier=(body.classifier if body else None),
                t5=(body.t5 if body else None),
            )
            result = analyze_pdf(norm, config=cfg)  # must return dict with "fields", "runtime", maybe "pages"
            title = os.path.basename(norm)
        else:
            if body is None:
                raise HTTPException(status_code=400, detail="Provide file_path or JSON body with tokens/bboxes.")
            cfg = _cfg_from_knobs(body.device, body.min_confidence, body.no_graph, body.classifier, body.t5)
            result = analyze_tokens(body.tokens, body.bboxes, page_ids=body.page_ids, config=cfg)
            title = "tokens.json"

        elapsed = time.time() - t0

        # Optional GT metrics (only if GT is available)
        gt = _load_gt_for(file_path or "")
        spans = None
        summar = None
        if gt:
            spans = evaluate_spans_iou(
                predicted=[{"label": f.get("label"), "bbox": f.get("bbox")} for f in result.get("fields", [])],
                ground_truth=[{"label": g.get("label"), "bbox": g.get("bbox")} for g in gt],
                iou_threshold=0.5,
            )
            refs = [g.get("reference_summary", "") for g in gt if g.get("reference_summary")]
            hyps = [f.get("summary", "") for f in result.get("fields", [])][:len(refs)]
            if refs and hyps:
                summar = compute_rouge_meteor(refs, hyps)

            # Optional detailed report
            report_text = format_metrics_for_report(classif=None, summar=summar, spans=spans)
            save_report_txt(report_text, path=METRICS_PATH)
        else:
            # Still write a quick operational report for the UI
            _write_quick_report(result, METRICS_PATH)

        # Build frontend-friendly response
        fields = result.get("fields", [])
        pages = result.get("pages") or result.get("num_pages")
        runtime = result.get("runtime", {})

        response = {
            "document": file_path or title,
            "title": title,
            "fields": fields,
            "metrics": {
                "precision": None,
                "recall": None,
                "f1": (spans or {}).get("f1"),
                "rougeL": (summar or {}).get("rougeL"),
                "meteor": (summar or {}).get("meteor"),
                "fields_count": len(fields),
                "processing_sec": elapsed,
                "pages": pages,
            },
            "runtime": runtime,
        }
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/metrics", response_class=PlainTextResponse)
def metrics():
    """Returns the latest quick metrics report if available."""
    if not os.path.exists(METRICS_PATH):
        return PlainTextResponse("No metrics report found.", status_code=404)
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return PlainTextResponse(f.read())

@app.get("/api/serve-file")
def serve_file(path: str = Query(..., description="Path to a file under static/ or uploads/")):
    norm = os.path.normpath(path)
    allowed_roots = [os.path.normpath(STATIC_DIR), os.path.normpath(UPLOAD_DIR)]
    if not any(norm.startswith(root) for root in allowed_roots):
        raise HTTPException(status_code=400, detail="Path must be under static/ or uploads/.")
    if not os.path.exists(norm):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(norm)
