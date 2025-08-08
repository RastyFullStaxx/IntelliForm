"""
IntelliForm — FastAPI App Entrypoint
====================================

WHAT THIS MODULE DOES
---------------------
Bootstraps the FastAPI application, mounts static files, registers routes,
and serves Jinja2 templates for the optional UI.

RESPONSIBILITIES
----------------
- Create FastAPI app instance.
- Mount `/static` -> ./static
- Register API routes from `api.py`.
- Render HTML templates:
    - GET /           -> index.html        (optional landing)
    - GET /workspace  -> workspace.html    (PDF viewer + sidebar UI)
- Provide a local development server via uvicorn.

TYPICAL DEV RUN
---------------
$ uvicorn main:app --host 0.0.0.0 --port 8000 --reload

INTERACTIONS
------------
- Uses: api.py (includes /api/* endpoints)
- Serves: templates/workspace.html, static assets (css/js/uploads)

DEPLOY NOTES
------------
- For production, run under a process manager (e.g., uvicorn + systemd, or Gunicorn+Uvicorn workers).
- Set proper static caching headers if needed.

TODOs
-----
- Add CORS configuration if UI is served from a different origin.
- Add graceful shutdown hooks for GPU resource cleanup (optional).

"""


from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from api import router as api_router
import os
import shutil
import traceback

# === FastAPI App Setup ===
app = FastAPI()

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 Use specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Static Assets & Templates ===
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# === Upload Directory ===
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === HTML Routes ===

@app.get("/", response_class=HTMLResponse)
async def render_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/workspace", response_class=HTMLResponse)
async def render_workspace(request: Request):
    return templates.TemplateResponse("workspace.html", {"request": request})

# === Upload Only (No Inference Here) ===

@app.post("/upload-pdf/", response_class=JSONResponse)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        filename = file.filename.replace(" ", "_")
        if not filename.lower().endswith(".pdf"):
            return JSONResponse(content={"error": "Only PDF files allowed"}, status_code=400)

        save_path = os.path.join(UPLOAD_FOLDER, filename)
        print(f"📥 PDF uploaded and saved to: {save_path}")

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return JSONResponse(content={"filename": filename})

    except Exception as e:
        print("🔥 Error during file upload:")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

# === Include API Routes (analysis, metrics, etc.) ===
app.include_router(api_router)
