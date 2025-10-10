# scripts/config.py
"""
IntelliForm — Config, Paths, Template Hashing, and Prelabeler Orchestration
===========================================================================

WHAT THIS MODULE DOES
---------------------
Centralizes:
- Environment switches (DEV/LIVE), logging level
- Canonical paths (uploads/, explanations/, _annotations/)
- Template-based hashing (robust to filled-in form values)
- Prelabeler launcher (subprocess) and simple promotion to canonical <HASH>.json
- LLM chat facade (kept neutral)

IMPORTANT CONVENTIONS
---------------------
- Frontend saves uploaded PDFs under: uploads/
- /api/upload (to be updated next) will return canonical_form_id = TEMPLATE_HASH
- Prelabeler writes a per-request temp annotation JSON: explanations/_annotations/<HASH>__temp.json
- Canonical annotation name: explanations/_annotations/<HASH>.json
- Explainer JSON (for UI sidebar) lives under explanations/<bucket>/<HASH>.json

DEPRECATION (TSL)
-----------------
- We no longer compare prelabel outputs to locate explainers. The canonical ID is
  the template-based hash. For API compatibility, `find_duplicate_annotations`
  and the old "promotion" orchestration remain but are no-ops (no matching).

NOTE
----
We intentionally avoid importing utils.llmv3_infer directly here to prevent
circular imports; we launch it as a subprocess (module-first, file-fallback).
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Union

# ------------------ Environment / Logging ------------------

def _read_bool_env(name: str, default: bool = False) -> bool:
    val = str(os.getenv(name, str(int(default)))).strip().lower()
    return val in {"1", "true", "yes", "y", "on"}

DEV_MODE: bool = _read_bool_env("INTELLIFORM_DEV_MODE", True)
LIVE_MODE: bool = _read_bool_env("INTELLIFORM_LLM_ENABLED", False)

# === Facade knobs (needed by utils/dual_head.py) ===
# Neutral secret name (no vendor wording)
CORE_ENGINE_KEY: str = (os.getenv("INTELLIFORM_CORE_KEY") or os.getenv("INTELLIFORM_ENGINE_KEY") or "").strip()
# Backend selector (kept neutral). Options: "oai" (default), "azure"
CORE_BACKEND: str = (os.getenv("INTELLIFORM_BACKEND") or "oai").strip().lower()
# Treat these like hyperparameters
ENGINE_MODEL: str = os.getenv("INTELLIFORM_ENGINE_MODEL", "gpt-4o-mini")
MAX_TOKENS: int = int(os.getenv("INTELLIFORM_ENGINE_MAXTOK", os.getenv("INTELLIFORM_MAX_TOKENS", "1200")) or 1200)
TEMPERATURE: float = float(os.getenv("INTELLIFORM_ENGINE_TEMP", os.getenv("INTELLIFORM_TEMPERATURE", "0.2")) or 0.2)

_LOG_LEVEL = os.getenv("INTELLIFORM_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("intelliform.config")

# ------------------ Paths ------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # project root (scripts/ under root)
UPLOADS_DIR = BASE_DIR / "uploads"                 # Frontend uses uploads/ (not static/uploads/)
EXPL_DIR = BASE_DIR / "explanations"
ANNO_DIR = EXPL_DIR / "_annotations"
REGISTRY_PATH = EXPL_DIR / "registry.json"         # /panel source

UTILS_DIR = BASE_DIR / "utils"
PRELABEL_ENTRY_FILE = UTILS_DIR / "llmv3_infer.py"

for _p in (UPLOADS_DIR, EXPL_DIR, ANNO_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ------------------ Constants ------------------

# Kept for compatibility; no longer used for matching logic.
DUPLICATE_THRESHOLD: float = float(os.getenv("INTELLIFORM_TSL_THRESHOLD", "0.75"))

# Per-request temp files: <form_id>__temp.json  (form_id is now the TEMPLATE HASH)
TEMP_ANNOTATION_SUFFIX = "__temp.json"

# ------------------ IDs / Sanitization ------------------

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")

def sanitize_form_id(name: str) -> str:
    """Stable, URL-safe-ish id (preserve underscores)."""
    stem = Path(name).stem
    stem = stem.strip().replace(" ", "-")
    stem = _SANITIZE_RE.sub("-", stem).strip("-")
    stem = re.sub(r"-{2,}", "-", stem)
    return stem

# ------------------ Template-based hash ------------------

_WS = re.compile(r"\s+")

def _norm_text(s: str) -> str:
    return _WS.sub(" ", (s or "").lower()).strip()

def _sha256_bytes(path: Union[str, Path]) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def template_hash_via_render(path: Union[str, Path]) -> Optional[str]:
    """
    Preferred: PyMuPDF (fitz) + perceptual hash per page with annots=False (widgets/typed values hidden).
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import imagehash
        import hashlib as _hl
        pdf_path = str(path)
        per_page = []
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(annots=False, alpha=False)  # ignore annotations/widgets
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            per_page.append(str(imagehash.phash(img)))
        doc.close()
        joined = "|".join(per_page)
        return _hl.sha256(joined.encode("utf-8")).hexdigest()
    except Exception as e:
        log.debug("template_hash_via_render failed: %s", e)
        return None

def template_hash_via_pypdf_text(path: Union[str, Path]) -> Optional[str]:
    """
    Fallback: strip AcroForm values (/V) & appearance (/AP) with pypdf; hash normalized text.
    """
    try:
        from pypdf import PdfReader
        import hashlib as _hl
        reader = PdfReader(str(path))
        root = reader.trailer.get("/Root", {})
        acro = root.get("/AcroForm", {})
        fields = acro.get("/Fields", [])
        for f in fields:
            try:
                if "/V" in f:
                    f["/V"] = ""
                if "/AP" in f:
                    del f["/AP"]
            except Exception:
                pass
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
        norm = _norm_text(" ".join(texts))
        if not norm:
            return None
        return _hl.sha256(norm.encode("utf-8")).hexdigest()
    except Exception as e:
        log.debug("template_hash_via_pypdf_text failed: %s", e)
        return None

def canonical_template_hash(path: Union[str, Path]) -> str:
    """
    Computes the TEMPLATE ID for a PDF (robust to filled-in values):
      1) Render pages with annotations disabled (perceptual page hash → SHA256)
      2) Else strip AcroForm values and hash normalized text
      3) Else SHA256(file bytes)
    """
    return (
        template_hash_via_render(path)
        or template_hash_via_pypdf_text(path)
        or _sha256_bytes(path)
    )
    
def extract_embedded_form_id(pdf_path: Union[str, Path]) -> Optional[str]:
    """
    Read PDF metadata and extract IntelliForm-embedded form_id, if any.
    We set this on the client export as:
        Subject: "IntelliForm-FormId:<hash>"
    """
    import re
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None

    try:
        with fitz.open(str(pdf_path)) as doc:
            subj = (doc.metadata or {}).get("subject") or ""
    except Exception:
        return None

    m = re.search(r"IntelliForm-FormId:([A-Za-z0-9_-]{8,128})", subj)
    return m.group(1) if m else None


# ------------------ Annotation paths ------------------

def temp_annotation_path(form_id: Optional[str] = None) -> Path:
    """
    Per-request temp annotation path to avoid collisions.
    If form_id is provided, use <form_id>__temp.json; else fall back to a session temp.
    NOTE: form_id is the TEMPLATE HASH.
    """
    if form_id:
        return ANNO_DIR / f"{form_id}{TEMP_ANNOTATION_SUFFIX}"
    return ANNO_DIR / ("_session" + TEMP_ANNOTATION_SUFFIX)

def canonical_annotation_path(form_id: str) -> Path:
    """form_id is the TEMPLATE HASH."""
    return ANNO_DIR / f"{form_id}.json"

# ------------------ Registry helpers ------------------

def get_registry_path() -> Path:
    return REGISTRY_PATH

def load_registry() -> Optional[Dict[str, Any]]:
    path = get_registry_path()
    if not path.exists():
        log.warning("Registry missing at %s", path)
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error("Failed to read registry %s: %s", path, e)
        return None

# ------------------ Prelabeler launcher ------------------

@dataclass
class LaunchResult:
    success: bool
    temp_annotation: Optional[Path]
    form_id: Optional[str]
    stderr: Optional[str] = None
    stdout: Optional[str] = None
    returncode: Optional[int] = None

def _python_exec() -> str:
    return sys.executable or "python"

def _run_prelabel_cli(pdf_path: Path, out_json: Path) -> subprocess.CompletedProcess:
    """
    llmv3_infer CLI:
      - module-first: python -m utils.llmv3_infer --prelabel --pdf <pdf> --out <temp_json> [--dev]
      - file fallback: python utils/llmv3_infer.py --prelabel ...
    """
    py = _python_exec()
    base_args = ["--prelabel", "--pdf", str(pdf_path), "--out", str(out_json)]
    if DEV_MODE:
        base_args.append("--dev")

    # Try module-first
    cmd_mod = [py, "-m", "utils.llmv3_infer", *base_args]
    log.info("Launching prelabeler (module): %s", " ".join(cmd_mod))
    mod_proc = subprocess.run(cmd_mod, capture_output=True, text=True)
    if mod_proc.returncode == 0:
        return mod_proc

    # Fallback: file path
    cmd_file = [py, str(PRELABEL_ENTRY_FILE), *base_args]
    log.info("Launching prelabeler (file): %s", " ".join(cmd_file))
    file_proc = subprocess.run(cmd_file, capture_output=True, text=True)
    return file_proc

def launch_prelabeler(
    pdf_path: Union[Path, str],
    out_temp: Optional[Union[Path, str]] = None,
    *,
    # Back-compat params (ignored but accepted)
    form_id: Optional[str] = None,
    out_path: Optional[Union[Path, str]] = None,
    repo_root: Optional[Union[Path, str]] = None,
) -> Union[LaunchResult, bool]:
    """
    Preferred (new) mode:
      launch_prelabeler(pdf_path: Path, out_temp: Path|None) -> LaunchResult

    Back-compat:
      launch_prelabeler(pdf_path=..., form_id=..., out_path=..., repo_root=...) -> bool
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        msg = f"PDF not found: {pdf_path}"
        log.error(msg)
        if out_path is not None:  # old mode expects bool
            return False
        return LaunchResult(False, None, None, stderr=msg, returncode=2)

    # Determine output path
    old_mode = out_path is not None
    out_json: Path = Path(out_path) if old_mode else (Path(out_temp) if out_temp else temp_annotation_path())
    out_json.parent.mkdir(parents=True, exist_ok=True)

    proc = _run_prelabel_cli(pdf_path, out_json)
    ok = out_json.exists()
    if not ok:
        log.error("Prelabeler failed. rc=%s stderr=%s", proc.returncode, proc.stderr)

    if old_mode:
        return bool(ok)

    form_id_guess = sanitize_form_id(pdf_path.name)
    return LaunchResult(
        success=bool(ok),
        temp_annotation=(out_json if ok else None),
        form_id=form_id_guess,
        stderr=proc.stderr,
        stdout=proc.stdout,
        returncode=proc.returncode,
    )

launch_prelabeler_compat = launch_prelabeler

# ------------------ (Deprecated) Duplicate matching — now a no-op ------------------

def find_duplicate_annotations(
    temp_anno_path: Union[Path, str],
    ann_dir: Optional[Union[Path, str]] = None,
    threshold: float = DUPLICATE_THRESHOLD,
) -> Tuple[Optional[Path], float]:
    """
    DEPRECATED: We no longer search for duplicates; canonical ID is the template hash.
    Kept for API compatibility. Always returns (None, 0.0).
    """
    return (None, 0.0)

@dataclass
class PromotionResult:
    success: bool
    reused: bool
    canonical_path: Optional[Path]
    match_score: float
    used_existing_path: Optional[Path] = None
    error: Optional[str] = None

def promote_or_reuse_annotation(
    form_id: str,
    temp_path: Optional[Union[Path, str]] = None,
    threshold: float = DUPLICATE_THRESHOLD,  # unused, kept for signature compat
) -> PromotionResult:
    """
    New behavior: simply promote the temp file to canonical <HASH>.json.
    No matching/TSL. `form_id` must be the TEMPLATE HASH.
    """
    temp = Path(temp_path) if temp_path else temp_annotation_path(form_id)
    if not temp.exists():
        return PromotionResult(False, False, None, 0.0, error=f"Temp annotation not found at {temp}")

    canonical = canonical_annotation_path(form_id)
    try:
        canonical.write_bytes(temp.read_bytes())
        try:
            temp.unlink()
        except Exception:
            pass
        return PromotionResult(True, False, canonical, match_score=1.0)
    except Exception as e:
        return PromotionResult(False, False, None, 0.0, error=f"Failed to promote temp → canonical: {e}")

# ------------------ High-level pipeline (for API wiring) ------------------

def run_prelabel_pipeline(
    pdf_path: Union[Path, str],
    *,
    form_id: Optional[str] = None,  # when None, a sanitized filename is used (legacy)
) -> Tuple[Optional[str], Optional[Path], Optional[str]]:
    """
    Launch prelabeler → produce _temp.json → promote to canonical.
    Returns: (form_id, canonical_path, error_msg)

    NOTE: In the new flow, API should pass the TEMPLATE HASH as form_id.
    """
    launch = launch_prelabeler(pdf_path)  # LaunchResult
    if isinstance(launch, bool):
        # legacy branch (should not be used in new flow)
        fid = form_id or sanitize_form_id(Path(pdf_path).name)
        if not launch:
            return (fid, None, "Prelabeler failed.")
        promo = promote_or_reuse_annotation(fid)
        if not promo.success or not promo.canonical_path:
            return (fid, None, promo.error or "Promotion failed.")
        return (fid, promo.canonical_path, None)

    if not launch.success or not launch.temp_annotation:
        err = launch.stderr or "Prelabeler failed without stderr."
        return (form_id or launch.form_id, None, err)

    fid = form_id or sanitize_form_id(Path(pdf_path).name)
    promo = promote_or_reuse_annotation(fid, launch.temp_annotation)
    if not promo.success or not promo.canonical_path:
        return (fid, None, promo.error or "Promotion failed.")
    return (fid, promo.canonical_path, None)

# ------------------ LLM chat facade ------------------

def chat_completion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    api_key: Optional[str] = None,
) -> str:
    """
    Return assistant text for a chat-style completion.
    Keeps vendor SDK details private to this module.
    """
    key = (api_key or CORE_ENGINE_KEY or "").strip()
    if not key:
        raise RuntimeError("Engine key missing.")

    backend = CORE_BACKEND

    if backend == "oai":
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("Core backend client unavailable.") from e
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    elif backend == "azure":
        endpoint = os.getenv("INTELLIFORM_AZURE_ENDPOINT", "").strip()
        deployment = os.getenv("INTELLIFORM_AZURE_DEPLOYMENT", "").strip()
        api_version = os.getenv("INTELLIFORM_AZURE_API_VERSION", "2024-06-01")
        if not endpoint or not deployment:
            raise RuntimeError("Azure backend misconfigured.")
        try:
            from openai import AzureOpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("Core backend client unavailable (azure).") from e
        client = AzureOpenAI(
            api_key=key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    else:
        raise RuntimeError("Unknown backend selected.")

# ------------------ Exports ------------------

__all__ = [
    # env / knobs
    "DEV_MODE", "LIVE_MODE",
    "CORE_ENGINE_KEY", "ENGINE_MODEL", "MAX_TOKENS", "TEMPERATURE",
    "CORE_BACKEND",
    # paths
    "BASE_DIR", "UPLOADS_DIR", "EXPL_DIR", "ANNO_DIR", "REGISTRY_PATH", "UTILS_DIR",
    # ids / hashing
    "sanitize_form_id",
    "template_hash_via_render", "template_hash_via_pypdf_text", "canonical_template_hash",
    # annotation paths
    "canonical_annotation_path", "temp_annotation_path",
    # registry
    "load_registry", "get_registry_path",
    # deprecated matching (no-op, kept for compat)
    "find_duplicate_annotations",
    # prelabel launcher / pipeline
    "launch_prelabeler", "launch_prelabeler_compat",
    "promote_or_reuse_annotation", "run_prelabel_pipeline",
    # dataclasses
    "LaunchResult", "PromotionResult",
    # chat
    "chat_completion",
    # ids / hashing
    "sanitize_form_id",
    "template_hash_via_render", "template_hash_via_pypdf_text", "canonical_template_hash",
    "extract_embedded_form_id",

]
