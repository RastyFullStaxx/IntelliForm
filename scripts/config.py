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


def _read_bool_env(name: str, default: bool = False) -> bool:
    val = str(os.getenv(name, str(int(default)))).strip().lower()
    return val in {"1", "true", "yes", "y", "on"}

DEV_MODE: bool = _read_bool_env("INTELLIFORM_DEV_MODE", True)
LIVE_MODE: bool = _read_bool_env("INTELLIFORM_LLM_ENABLED", True)

CORE_ENGINE_KEY: str = (os.getenv("INTELLIFORM_CORE_KEY") or os.getenv("INTELLIFORM_ENGINE_KEY") or "").strip()
CORE_BACKEND: str = (os.getenv("INTELLIFORM_BACKEND") or "oai").strip().lower()
ENGINE_MODEL: str = os.getenv("INTELLIFORM_ENGINE_MODEL", "gpt-4o-mini")
MAX_TOKENS: int = int(os.getenv("INTELLIFORM_ENGINE_MAXTOK", os.getenv("INTELLIFORM_MAX_TOKENS", "6000")) or 6000)
TEMPERATURE: float = float(os.getenv("INTELLIFORM_ENGINE_TEMP", os.getenv("INTELLIFORM_TEMPERATURE", "0.2")) or 0.2)
ENGINE_KEY_HINT = "env > file > INLINE_FALLBACK_KEY"

_LOG_LEVEL = os.getenv("INTELLIFORM_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("intelliform.config")

"""
  BASELINE SWITCH. FLIP LINE 40 TO FALSE TO DISABLE and vice versa
"""
BASELINE_MODE: bool = _read_bool_env("INTELLIFORM_BASELINE_MODE", False)
BASELINE_BACKEND: str = (os.getenv("INTELLIFORM_BASELINE_BACKEND", "llm") or "llm").strip().lower()
BASELINE_DROP_RATE: float = float(os.getenv("INTELLIFORM_BASELINE_DROP_RATE", "0.35"))
BASELINE_MISLABEL_RATE: float = float(os.getenv("INTELLIFORM_BASELINE_MISLABEL_RATE", "0.15"))
BASELINE_VAGUE_RATE: float = float(os.getenv("INTELLIFORM_BASELINE_VAGUE_RATE", "0.60"))
BASELINE_SEED = (int(os.getenv("INTELLIFORM_BASELINE_SEED", "").strip())
                 if os.getenv("INTELLIFORM_BASELINE_SEED") not in (None, "",) else None)
BASELINE_HINT = "baseline switch for A/B flow"

"""
  Fallback inline key (for dev/testing)
"""
INLINE_FALLBACK_KEY: str = ""

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


UPLOADS_DIR = BASE_DIR / "uploads"                 # Frontend uses uploads/ (not static/uploads/)
EXPL_DIR = BASE_DIR / "explanations"
ANNO_DIR = EXPL_DIR / "_annotations"
REGISTRY_PATH = EXPL_DIR / "registry.json"         # /panel source

UTILS_DIR = BASE_DIR / "utils"
PRELABEL_ENTRY_FILE = UTILS_DIR / "llmv3_infer.py"

for _p in (UPLOADS_DIR, EXPL_DIR, ANNO_DIR, SECRETS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


EPHEMERAL_ANNOS: bool = _read_bool_env("INTELLIFORM_EPHEMERAL_ANNOS", True)
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


DUPLICATE_THRESHOLD: float = float(os.getenv("INTELLIFORM_TSL_THRESHOLD", "0.75"))

TEMP_ANNOTATION_SUFFIX = "__temp.json"


_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")

def sanitize_form_id(name: str) -> str:
    """Stable, URL-safe-ish id (preserve underscores)."""
    stem = Path(name).stem
    stem = stem.strip().replace(" ", "-")
    stem = _SANITIZE_RE.sub("-", stem).strip("-")
    stem = re.sub(r"-{2,}", "-", stem)
    return stem


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
    
def quick_text_snippet(pdf_path: str, max_chars: int = 4000) -> str:
    """Extract visible page text via PyMuPDF; return a capped snippet."""
    try:
        import fitz  # PyMuPDF
        acc = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                t = page.get_text("text") or ""
                if t.strip():
                    acc.append(t.strip())
                if sum(len(x) for x in acc) > max_chars:
                    break
        blob = "\n".join(acc)
        return blob[:max_chars]
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

    cmd_mod = [py, "-m", "utils.llmv3_infer", *base_args]
    log.info("Launching prelabeler (module): %s", " ".join(cmd_mod))
    mod_proc = subprocess.run(cmd_mod, capture_output=True, text=True)
    if mod_proc.returncode == 0:
        return mod_proc

    cmd_file = [py, str(PRELABEL_ENTRY_FILE), *base_args]
    log.info("Launching prelabeler (file): %s", " ".join(cmd_file))
    file_proc = subprocess.run(cmd_file, capture_output=True, text=True)
    return file_proc

def launch_prelabeler(
    pdf_path: Union[Path, str],
    out_temp: Optional[Union[Path, str]] = None,
    *,
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
- Use neighboring cues (row/column headers, section titles, nearby descriptors) to keep summaries specific while staying faithful to visible text; do not invent values.
- Prefer the exact printed label text as it appears on the form for every field label.
  • Trim extra punctuation and trailing colons.
  • Preserve casing & words so our click-to-jump overlay can match reliably.
- DO NOT merge two distinct printed labels into one field, unless the form itself prints them as a single combined label.
- Group fields by on-page layout order, scanning left→right, then top→bottom.
- Include EVERY question/blank on the form that requires a user answer AND include header/top banners, corner boxes, and “For Official Use” areas (set summary like “Office use only; leave blank” if applicable).
- Keep summaries short, imperative, and specific (what to write/tick); ensure the instruction matches the label’s intent (no vague wording).
- If the printed field shows placeholder text like "Type here" or "Click here to enter a date", do NOT copy it. Instead, state the expected user input inferred from the label (e.g., "Enter full birth date (MM/DD/YYYY, e.g., 11/28/2025)", "Enter complete present address").
- Use the form’s title/heading to infer the document’s purpose so instructions feel context-aware (e.g., learner application vs. tax form); still avoid hallucinating unseen fields.
- For dates or similar formats, ALWAYS include the format and a concrete example in the summary (e.g., “Use MM/DD/YYYY (e.g., 11/28/2025)”).
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
1) Read the page text (OCR if needed). Extract all answer-requiring labels, including header/corner boxes and “Official Use Only” rows.
2) Use exact visible labels (trim punctuation/colons). One printed label → one field.
3) Group by layout (left→right, top→bottom) into concise sections; use nearby headings/row titles to keep summaries context-aware without making up content.
4) Write short imperative summaries describing what the user should write or tick; replace placeholder prompts ("Type here", "Click to enter date") with the actual expected input based on the label and form purpose. For ANY date fields, state the format AND an explicit example (e.g., "Use MM/DD/YYYY (e.g., 11/28/2025)").
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
        return '{"error":"no_engine_key"}'

    try:
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
        return json.dumps({"error":"engine_call_failed","detail":str(e)})


__all__ = [
    "DEV_MODE", "LIVE_MODE",
    "CORE_ENGINE_KEY", "ENGINE_MODEL", "MAX_TOKENS", "TEMPERATURE",
    "CORE_BACKEND",
    "BASE_DIR", "UPLOADS_DIR", "EXPL_DIR", "ANNO_DIR", "REGISTRY_PATH", "UTILS_DIR",
    "sanitize_form_id",
    "template_hash_via_render", "template_hash_via_pypdf_text", "canonical_template_hash",
    "extract_embedded_form_id",
    "canonical_annotation_path", "temp_annotation_path",
    "load_registry", "get_registry_path",
    "find_duplicate_annotations",
    "launch_prelabeler", "launch_prelabeler_compat",
    "promote_or_reuse_annotation", "run_prelabel_pipeline",
    "LaunchResult", "PromotionResult",
    "chat_completion",
    "EXPLAINER_SYSTEM_PROMPT", "EXPLAINER_USER_PROMPT_TEMPLATE", "EXPLAINER_SCHEMA_NOTES",
    "build_explainer_messages",
    "EPHEMERAL_ANNOS", "ANNO_TTL_SECONDS", "purge_stale_annotations",
    "ENGINE_KEY_FILE", "INLINE_FALLBACK_KEY",
    "quick_text_snippet",
    "build_explainer_messages_with_context",
        "BASELINE_MODE", "BASELINE_BACKEND",
    "BASELINE_DROP_RATE", "BASELINE_MISLABEL_RATE", "BASELINE_VAGUE_RATE",
    "BASELINE_SEED",
]
