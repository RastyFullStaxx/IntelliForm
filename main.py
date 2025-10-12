# main.py
"""
IntelliForm — FastAPI App Entrypoint
====================================

WHAT THIS MODULE DOES
---------------------
Bootstraps the FastAPI application, mounts static/uploads, registers UI routes,
and reuses the API app defined in `api.py`.

RESPONSIBILITIES
----------------
- Import the FastAPI app from `api.py` (all /api/* and /panel routes are ready)
- Mount `/uploads` to serve user PDFs directly to the PDF.js viewer
- Serve Jinja2 templates for the UI:
    - GET /           -> index.html (optional; falls back to workspace.html)
    - GET /workspace  -> workspace.html (central UI)

TYPICAL DEV RUN
---------------
$ python main.py
  or
$ uvicorn main:app --host 127.0.0.1 --port 8000 --reload

INTERACTIONS
------------
- Uses: api.py (provides /api/*, /panel, and static mounts for /static and /explanations)
- Serves: templates/*.html and static assets for the workspace
"""

# main.py
from __future__ import annotations
import os

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

# Neutral runtime knobs (host/port/loglevel) come from config, if present
try:
    from scripts.config import get_mode_string  # optional, for logging
except Exception:
    def get_mode_string() -> str:
        return "pipeline"

# Reuse API app (api.py exports `app`)
from api import app as app  # FastAPI instance

# Directories
STATIC_DIR    = "static"        # already mounted in api.py
EXPL_DIR      = "explanations"  # already mounted in api.py
UPLOADS_DIR   = "uploads"
TEMPLATES_DIR = "templates"

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Mount uploads here (avoid double-mounting /static and /explanations)
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Render templates/index.html if present; otherwise workspace.html.
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        return templates.TemplateResponse("index.html", {"request": request})
    return templates.TemplateResponse("workspace.html", {"request": request})

@app.get("/workspace", response_class=HTMLResponse)
async def workspace(request: Request):
    return templates.TemplateResponse("workspace.html", {"request": request})

if __name__ == "__main__":
    import uvicorn

    # Read neutral runtime options from env (or leave defaults)
    host = os.getenv("INTELLIFORM_HOST", "127.0.0.1")
    port = int(os.getenv("INTELLIFORM_PORT", "8000") or 8000)
    log_level = os.getenv("INTELLIFORM_LOG_LEVEL", "warning")

    # Optional: tiny banner without exposing internals
    try:
        mode_str = get_mode_string()
        print(f"[IntelliForm] UI ready on http://{host}:{port}  |  mode: {mode_str}")
    except Exception:
        pass

    uvicorn.run("main:app", host=host, port=port, reload=True, log_level=log_level)

@app.get("/favicon.ico")
async def favicon():
    # Serve the PNG as the canonical favicon for agents that request /favicon.ico
    return RedirectResponse(url="/static/img/imgMainLogo.png")