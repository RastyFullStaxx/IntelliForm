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

PRIMARY ENDPOINTS (SUGGESTED)
-----------------------------
- GET  /api/health
    Returns {"status": "ok"} for liveness checks.

- POST /api/upload
    Multipart upload of a PDF. Saves to `uploads/` and returns a file_id/path.

- POST /api/analyze
    Body: {"pdf_path": "..."} or {"file_id": "..."} plus optional config.
    Runs full pipeline: extract -> classify -> group -> summarize.
    Returns structured JSON with fields, bboxes, confidences, summaries, timing.

- GET  /api/metrics
    Returns latest metrics report (if produced during evaluation runs), or
    triggers a quick evaluation on a small dev set (optional).

- GET  /api/serve-file?path=...
    Serves generated artifacts like `static/metrics_report.txt` if needed.

RESPONSE SHAPE (ANALYZE)
------------------------
{
  "document": "uploads/2025-08-08_21-30_form.pdf",
  "fields": [
    {"label": "FULL_NAME", "score": 0.93, "bbox": [x0,y0,x1,y1], "summary": "...", "page": 1},
    ...
  ],
  "runtime": {"extract_ms": 120, "classify_ms": 85, "summarize_ms": 40}
}

INTERACTIONS
------------
- Called by: frontend (templates/workspace.html + static/js/workspace.js) or external clients
- Calls into: utils.llmv3_infer.analyze_pdf / analyze_tokens

SECURITY / DEPLOY NOTES
-----------------------
- Validate file paths; restrict to `uploads/`.
- Enforce file size/type limits.
- Consider async endpoints for long-running analyses.

TODOs
-----
- Add auth if exposing beyond localhost.
- Add background tasks for large multi-page PDFs.

"""


import os
import shutil
from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import JSONResponse
from inference import run_inference
from utils.metrics import compute_dummy_ece_score, save_analysis_to_txt

# 🚩 Router initialization
router = APIRouter()

# 📁 Directory to save uploaded PDFs
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 📤 Upload endpoint — saves file only, no inference triggered
@router.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename.replace(" ", "_")
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"📥 PDF uploaded and saved to: {filepath}")
    return {"filename": filename}

# 🧠 Optional endpoint: Immediate analyze on upload (not used by frontend)
@router.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    filename = file.filename.replace(" ", "_")
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        results = run_inference(filepath)
        ece_score = compute_dummy_ece_score(results)
        save_analysis_to_txt(results, ece_score)

        return JSONResponse(content={
            "results": results,
            "ece": ece_score
        })

    except Exception as e:
        print(f"❌ Error during analyze_pdf: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# 🔁 Main endpoint: Analyze a previously uploaded file (used by frontend)
@router.get("/analyze-saved")
async def analyze_saved(file: str = Query(...)):
    filepath = os.path.join(UPLOAD_DIR, file)
    print(f"📥 [API] Received analyze-saved request for file: {file}")

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return JSONResponse(status_code=404, content={"error": "File not found."})

    try:
        results = run_inference(filepath)
        ece_score = compute_dummy_ece_score(results)
        save_analysis_to_txt(results, ece_score)

        print(f"📤 [API] Returning {len(results)} results with ECE: {ece_score:.4f}")
        return JSONResponse(content={
            "results": results,
            "ece": ece_score
        })

    except Exception as e:
        print(f"❌ Exception in analyze-saved: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

