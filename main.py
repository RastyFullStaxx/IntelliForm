# main.py

"""
IntelliForm — FastAPI App Entrypoint
====================================

WHAT THIS MODULE DOES
---------------------
Bootstraps the FastAPI application, mounts static files, registers UI routes,
and reuses the API app defined in `api.py`.

RESPONSIBILITIES
----------------
- Import the FastAPI app from `api.py` (so /api/* is already wired)
- Mount `/static` -> ./static
- Serve Jinja2 templates for the optional UI:
    - GET /           -> index.html (optional landing; falls back to workspace)
    - GET /workspace  -> workspace.html (central UI)

TYPICAL DEV RUN
---------------
$ uvicorn main:app --host 0.0.0.0 --port 8000 --reload

INTERACTIONS
------------
- Uses: api.py (provides /api/* endpoints and CORS)
- Serves: templates/workspace.html, static assets (css/js/uploads)

DEPLOY NOTES
------------
- In production, run behind a process manager (e.g., Gunicorn + Uvicorn workers).
- Tighten CORS in `api.py` and add caching headers for static if needed.
"""

from __future__ import annotations
import os

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Reuse the API app (already includes /api/*, CORS, and directories)
from api import app as api_app

# Alias to the app variable expected by uvicorn (uvicorn main:app)
app = api_app

# --- Static & Templates ---
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# --- UI Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Optional landing page. If templates/index.html is missing,
    render workspace.html instead.
    """
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        return templates.TemplateResponse("index.html", {"request": request})
    # Fallback to workspace if index is absent
    return templates.TemplateResponse("workspace.html", {"request": request})


@app.get("/workspace", response_class=HTMLResponse)
async def workspace(request: Request):
    """
    Main UI page (PDF viewer + results panel).
    """
    return templates.TemplateResponse("workspace.html", {"request": request})


# --- Local Dev Runner ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
