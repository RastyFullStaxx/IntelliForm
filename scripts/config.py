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
- Prelabeler launcher (subprocess) and promotion/cleanup policy
- LLM chat facade (kept vendor-neutral) + prompt builders for Explainers

KEY UPDATES (2025-10-11)
------------------------
1) Engine key persistence:
   - Precedence: ENV > secrets file > inline fallback constant.
   - Put your key once into `INLINE_FALLBACK_KEY` or create the secrets file:
       scripts/.secrets/engine.key
   - On new machines, you won't need to paste it again if the file/inline stays.

2) Ephemeral annotations (optional):
   - EPHEMERAL_ANNOS toggles whether we keep canonical <HASH>.json on disk.
   - If True (session/temporary behavior), we still write canonical files so
     the frontend fetch path stays stable, but we auto-clean files older than
     ANNO_TTL_SECONDS on import and on each promotion.

3) Explainer prompts:
   - System & user prompt templates that require:
       • exact visual labels where possible (for click-to-jump),
       • no hallucinations (use OCR; say “N/A” or “—” if absent),
       • don’t merge two distinct labels unless the visual form merges them,
       • left→right, top→bottom grouping into accordion sections,
       • realistic per-PDF metrics (precision/recall/F1) that reflect model
         confidence,
       • canonical metadata (title, bucket, canonical_id, aliases),
       • schema matches the UI (title/sections/fields/metrics/...).

IMPORTANT CONVENTIONS
---------------------
- Frontend saves uploaded PDFs under: uploads/
- /api/upload returns canonical_template_hash as canonical_form_id (TEMPLATE HASH)
- Prelabeler writes a per-request temp JSON: explanations/_annotations/<HASH>__temp.json
- Canonical annotation name: explanations/_annotations/<HASH>.json
- Explainer JSON (for sidebar) lives under explanations/<bucket>/<HASH>.json
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import time
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

# ------------------ Secrets & Key persistence ------------------

# 1) Place a plaintext key here (inline fallback). Safer than envs on throwaway boxes.
#    Example: INLINE_FALLBACK_KEY = "sk-xxxxx"
INLINE_FALLBACK_KEY: str = ""  # <== paste once here if you prefer

# 2) Or put a file at scripts/.secrets/engine.key with your key
BASE_DIR = Path(__file__).resolve().parent.parent  # project root (scripts/ under root)
SECRETS_DIR = BASE_DIR / "scripts" / ".secrets"
ENGINE_KEY_FILE = SECRETS_DIR / "engine.key"

def _load_key_from_file() -> str:
    try:
        if ENGINE_KEY_FILE.exists():
            key = ENGINE_KEY_FILE.read_text(encoding="utf-8").strip()
            if key:
                return key
    except Exception as e:
        log.warning("Failed reading engine key file: %s", e)
    return ""

def _ensure_engine_key():
    global CORE_ENGINE_KEY
    if CORE_ENGINE_KEY:
        return
    key = _load_key_from_file() or INLINE_FALLBACK_KEY.strip()
    if key:
        CORE_ENGINE_KEY = key
        log.info("Engine key loaded via %s",
                 "file" if ENGINE_KEY_FILE.exists() else "inline fallback")
    else:
        log.error("Engine key missing. Set ENV, create %s, or paste INLINE_FALLBACK_KEY.", ENGINE_KEY_FILE)

_ensure_engine_key()

# ------------------ Paths ------------------

UPLOADS_DIR = BASE_DIR / "uploads"                 # Frontend uses uploads/ (not static/uploads/)
EXPL_DIR = BASE_DIR / "explanations"
ANNO_DIR = EXPL_DIR / "_annotations"
REGISTRY_PATH = EXPL_DIR / "registry.json"         # /panel source

UTILS_DIR = BASE_DIR / "utils"
PRELABEL_ENTRY_FILE = UTILS_DIR / "llmv3_infer.py"

for _p in (UPLOADS_DIR, EXPL_DIR, ANNO_DIR, SECRETS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ------------------ Ephemeral annotations (optional) ------------------

# If True: keep canonical files for a short time only (so frontend path is stable)
# and purge old ones automatically.
EPHEMERAL_ANNOS: bool = _read_bool_env("INTELLIFORM_EPHEMERAL_ANNOS", False)
ANNO_TTL_SECONDS: int = int(os.getenv("INTELLIFORM_ANNO_TTL_SECONDS", "7200"))  # 2 hours default

def purge_stale_annotations(now: Optional[float] = None) -> int:
    """Delete canonical and temp annotation files older than TTL. Returns count purged."""
    if not EPHEMERAL_ANNOS:
        return 0
    now = now or time.time()
    purged = 0
    try:
        for p in ANNO_DIR.glob("*.json"):
            try:
                age = now - p.stat().st_mtime
                if age > ANNO_TTL_SECONDS:
                    p.unlink(missing_ok=True)
                    purged += 1
            except Exception:
                pass
    except Exception as e:
        log.warning("purge_stale_annotations: %s", e)
    if purged:
        log.info("Purged %d stale annotation(s).", purged)
    return purged

purge_stale_annotations()

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
    Subject pattern: "IntelliForm-FormId:<hash>"
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
    
# --- REPLACE quick_text_snippet in scripts/config.py ---

def quick_text_snippet(pdf_path: str, max_chars: int = 12000) -> str:
    """
    Extract visible text. Strategy:
      1) Fast: PyMuPDF page text
      2) Fallback (opt-in via INTELLIFORM_OCR_SNIPPET=1): OCR first few pages (dpi=180)
    """
    try:
        import fitz  # PyMuPDF
        acc = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                t = page.get_text("text") or ""
                if t.strip():
                    acc.append(t.strip())
                if sum(len(x) for x in acc) >= max_chars:
                    break
        blob = "\n".join(acc).strip()
        if blob:
            return blob[:max_chars]
    except Exception:
        pass

    # Optional OCR fallback
    use_ocr = str(os.getenv("INTELLIFORM_OCR_SNIPPET", "0")).lower() in {"1", "true", "yes"}
    if not use_ocr:
        return ""

    try:
        import fitz
        from PIL import Image
        import pytesseract
        chunks = []
        with fitz.open(pdf_path) as doc:
            pages = min(3, len(doc))  # OCR first few pages only
            for pno in range(pages):
                pm = doc[pno].get_pixmap(dpi=180)  # decent speed/quality balance
                im = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                chunks.append(pytesseract.image_to_string(im))
        ocr_txt = "\n".join(x.strip() for x in chunks if x and x.strip())
        return ocr_txt[:max_chars]
    except Exception:
        return ""

def build_explainer_messages_with_context(
    *, canonical_id: str, bucket_guess: str, title_guess: str,
    text_snippet: str, candidate_labels: list[str] | None = None
):
    """
    Stricter, context-rich messages that enforce 'verbatim-only' labels.
    """
    sys = (
        "You are IntelliForm Explainer. Output strictly ONE JSON object. "
        "NO markdown fences and NO commentary. Use ONLY information that is explicitly present "
        "in the provided text excerpt (verbatim OCR/visible text from the PDF). "
        "EVERY field label in your JSON must be a substring of the provided text "
        "(case-insensitive, whitespace-normalized). If a label is not present, OMIT it. "
        "Do NOT invent fields or sections. Prefer the exact printed wording for labels; "
        "trim trailing colons/punctuation."
    )

    rules = EXPLAINER_SCHEMA_NOTES.strip()

    labels_hint = ""
    if candidate_labels:
        uniq = sorted({(l or "").strip() for l in candidate_labels if l})
        if uniq:
            labels_hint = "Candidate field labels (from detector/annotations):\n- " + "\n- ".join(uniq[:120]) + "\n"

    user = f"""
Generate the explainer JSON for a PDF form.

Canonical template hash (ID): {canonical_id}
Bucket guess: {bucket_guess}
Title guess: {title_guess}

Use ONLY what can be justified from this text (verbatim OCR excerpt):
---
{text_snippet}
---

{labels_hint}
Schema & strict rules:
{rules}

Output ONLY the JSON.
""".strip()

    return [{"role":"system","content":sys},{"role":"user","content":user}]

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
    New behavior: promote temp file to canonical <HASH>.json (frontend expects this path).
    If EPHEMERAL_ANNOS is True, we still write canonical so the UI works, but schedule
    cleanup by TTL. Temp file is always deleted if promotion succeeds.
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
        if EPHEMERAL_ANNOS:
            purge_stale_annotations()  # best-effort periodic cleanup
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

    NOTE: API should pass TEMPLATE HASH as form_id when available.
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

# ------------------ LLM Explainer prompts ------------------
# These builders produce messages for a chat completion that yields the
# sidebar Explainer JSON. It assumes the render/OCR pipeline provides text.

EXPLAINER_SCHEMA_NOTES = """
You are generating a JSON explainer for a PDF form. Output MUST be valid JSON with the shape:
{
  "title": <string>,
  "form_id": <string>,            // human-readable id or slug
  "sections": [
    {
      "title": <string>,
      "fields": [
        {"label": <string>, "summary": <string>},
        ...
      ]
    },
    ...
  ],
  "metrics": {
    "tp": <int>, "fp": <int>, "fn": <int>,
    "precision": <float>, "recall": <float>, "f1": <float>
  },
  "canonical_id": <string>,       // TEMPLATE HASH
  "bucket": <string>,             // e.g., government | banking | tax | healthcare
  "schema_version": 1,
  "created_at": <ISO8601>,
  "updated_at": <ISO8601>,
  "aliases": [<string>...]
}

STRICT RULES:
- NO hallucinations. If a required label cannot be verified from the page text, set its summary to "N/A" or "—".
- Prefer the exact printed label text as it appears on the form for every field label.
  • Trim extra punctuation and trailing colons.
  • Preserve casing & words so our click-to-jump overlay can match reliably.
- DO NOT merge two distinct printed labels into one field, unless the form itself prints them as a single combined label.
- Group fields by on-page layout order, scanning left→right, then top→bottom.
- Include EVERY question/blank on the form that requires a user answer.
- Keep summaries short, imperative, and specific (what to write/tick).
- If OCR text is noisy, normalize whitespace and obvious OCR artifacts; still do not invent content.
- Provide realistic metrics (tp/fp/fn → precision/recall/f1) that reflect your own confidence on THIS document:
  • High confidence on clearly read labels (sharp print, unambiguous layout) → higher precision/recall.
  • If any pages are low quality/ambiguous, lower the metrics accordingly.
- Keep floats to 3 decimals (e.g., 0.872). Compute f1 = 2 * P * R / (P + R) with the same rounding.
- Titles and section titles should reflect visible headings or logical groupings on the page.
- Aliases may include filename-like variants or common product names if visible.
"""

EXPLAINER_SYSTEM_PROMPT = (
    "You are IntelliForm’s document explainer. "
    "You extract precise, non-hallucinated field labels and instructions from PDFs. "
    "You optimize labels for programmatic matching to overlay boxes. "
    "When uncertain, you mark items as 'N/A' rather than guessing. "
    "You follow the schema and rules exactly."
)

EXPLAINER_USER_PROMPT_TEMPLATE = """
Generate the explainer JSON for a PDF form.

Context:
- Canonical template hash (ID): {canonical_id}
- Bucket guess: {bucket_guess}
- Title guess: {title_guess}

Tasks:
1) Read the page text (OCR if needed). Extract all answer-requiring labels.
2) Use exact visible labels (trim punctuation/colons). One printed label → one field.
3) Group by layout (left→right, top→bottom) into concise sections.
4) Write short imperative summaries describing what the user should write or tick.
5) Fill the metadata (canonical_id, bucket, schema_version=1, aliases if applicable).
6) Produce realistic metrics for THIS file reflecting your own certainty.
7) Output ONLY the JSON. No prose.

Additional schema constraints and rules:
{schema_rules}
""".strip()

def build_explainer_messages(
    *,
    canonical_id: str,
    bucket_guess: str,
    title_guess: str,
) -> List[Dict[str, Any]]:
    """Return messages=[{role,content}...] to send to the chat backend for explainer generation."""
    user = EXPLAINER_USER_PROMPT_TEMPLATE.format(
        canonical_id=canonical_id,
        bucket_guess=bucket_guess,
        title_guess=title_guess,
        schema_rules=EXPLAINER_SCHEMA_NOTES.strip(),
    )
    return [
        {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

# ------------------ LLM chat facade ------------------

# --- replace ONLY this function in scripts/config.py ---

def chat_completion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    api_key: Optional[str] = None,
    enforce_json: bool = False,  # <-- NEW
) -> str:
    """
    Return assistant text for a chat-style completion.
    When enforce_json=True, use response_format={'type':'json_object'} so we get a strict JSON object.
    Do NOT parse here; callers (api/dual_head) will sanitize+parse.
    """
    key = (api_key or CORE_ENGINE_KEY or "").strip()
    if not key:
        # Let caller fall back to scaffold rather than exploding here
        return '{"error":"no_engine_key"}'

    try:
        # Optional debug flag: INTELLIFORM_LLM_DEBUG=1
        LLM_DEBUG = str(os.getenv("INTELLIFORM_LLM_DEBUG", "0")).strip().lower() in {"1","true","yes"}

        def _dump(tag: str, text: str):
            if not LLM_DEBUG:
                return
            out_dir = BASE_DIR / "out" / "_llm"
            out_dir.mkdir(parents=True, exist_ok=True)
            import time as _t
            (out_dir / f"{int(_t.time())}_{tag}.txt").write_text(text, encoding="utf-8")

        if CORE_BACKEND == "oai":
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=key)
            kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
            if enforce_json:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            _dump("RAW", text)
            return text

        elif CORE_BACKEND == "azure":
            from openai import AzureOpenAI  # type: ignore
            endpoint   = os.getenv("INTELLIFORM_AZURE_ENDPOINT", "").strip()
            deployment = os.getenv("INTELLIFORM_AZURE_DEPLOYMENT", "").strip()
            api_version= os.getenv("INTELLIFORM_AZURE_API_VERSION", "2024-06-01")
            if not endpoint or not deployment:
                return '{"error":"azure_misconfigured"}'
            client = AzureOpenAI(api_key=key, api_version=api_version, azure_endpoint=endpoint)
            kwargs = dict(model=deployment, messages=messages, max_tokens=max_tokens, temperature=temperature)
            if enforce_json:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            _dump("RAW_AZ", text)
            return text

        else:
            return '{"error":"unknown_backend"}'

    except Exception as e:
        # Let caller’s try/except decide to scaffold
        return json.dumps({"error":"engine_call_failed","detail":str(e)})

# --- ADD in scripts/config.py (below chat_completion) ---

def generate_explainer_json_strict(
    *,
    canonical_id: str,
    bucket_guess: str,
    title_guess: str,
    text_snippet: str,
    candidate_labels: Optional[List[str]] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    log_dir: Optional[Path] = None,
) -> str:
    """
    Builds context-rich messages and calls the chat backend with:
      - response_format=json_object
      - one retry if parsing/sections empty
      - optional raw dump logs
    Returns a JSON string (may be a scaffold-like minimal JSON if both tries fail).
    """
    m = model or ENGINE_MODEL
    mt = max_tokens or MAX_TOKENS
    tmp = temperature if temperature is not None else TEMPERATURE

    messages = build_explainer_messages_with_context(
        canonical_id=canonical_id,
        bucket_guess=bucket_guess,
        title_guess=title_guess,
        text_snippet=text_snippet,
        candidate_labels=candidate_labels,
    )

    def _dump_raw(tag: str, payload: str):
        try:
            if not log_dir:
                return
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{canonical_id}.{tag}.json").write_text(payload, encoding="utf-8")
        except Exception:
            pass

    # try 1
    raw1 = chat_completion(
        model=m, messages=messages, max_tokens=mt, temperature=tmp, enforce_json=True
    )
    _dump_raw("try1", raw1)

    def _parse_js(s: str) -> Dict[str, Any]:
        try:
            return json.loads(s)
        except Exception:
            return {}

    data1 = _parse_js(raw1)
    if (data1.get("sections") or []) and isinstance(data1.get("sections"), list):
        return raw1

    # try 2 with extra nudge
    messages2 = messages + [{"role": "system", "content": "Return strict JSON only (no prose)."}]
    raw2 = chat_completion(
        model=m, messages=messages2, max_tokens=mt, temperature=tmp, enforce_json=True
    )
    _dump_raw("try2", raw2)

    data2 = _parse_js(raw2)
    if (data2.get("sections") or []) and isinstance(data2.get("sections"), list):
        return raw2

    # last resort: minimal scaffold; caller can fill metadata/metrics
    return json.dumps({
        "title": title_guess,
        "form_id": sanitize_form_id(title_guess),
        "sections": [],
        "metrics": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
        "canonical_id": canonical_id,
        "bucket": bucket_guess,
        "schema_version": 1,
        "created_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "updated_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "aliases": []
    }, ensure_ascii=False)

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
    "extract_embedded_form_id",
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
    "EXPLAINER_SYSTEM_PROMPT", "EXPLAINER_USER_PROMPT_TEMPLATE", "EXPLAINER_SCHEMA_NOTES",
    "build_explainer_messages",
    # ephemeral controls
    "EPHEMERAL_ANNOS", "ANNO_TTL_SECONDS", "purge_stale_annotations",
    "ENGINE_KEY_FILE", "INLINE_FALLBACK_KEY",
    "quick_text_snippet",
    "build_explainer_messages_with_context",
    # add to __all__ list:
    "generate_explainer_json_strict",
]
