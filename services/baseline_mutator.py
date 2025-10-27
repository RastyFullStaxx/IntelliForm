# services/baseline_mutator.py
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
    
from scripts import config

# ---- Small helpers ----

VALID_BUCKETS = {"government", "banking", "tax", "healthcare"}

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _deepcopy(x: Any) -> Any:
    return json.loads(json.dumps(x, ensure_ascii=False))

def _norm_rate(v: Optional[float], default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        f = float(v if v is not None else default)
        return max(lo, min(hi, f))
    except Exception:
        return max(lo, min(hi, default))

def _rng(seed: Optional[int]) -> random.Random:
    if seed is None:
        return random.Random()
    r = random.Random()
    r.seed(seed)
    return r

# ---- Canonical explainer resolution (read-only) ----

def _load_registry() -> Dict[str, Any]:
    path = config.get_registry_path()
    if not path.exists():
        return {"forms": []}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f) or {"forms": []}
    except Exception:
        return {"forms": []}

def _find_explainer_path(canonical_id: str) -> Optional[Path]:
    """
    Try registry; fall back to scanning explanations/<bucket>/<id>.json.
    """
    rid = (canonical_id or "").strip()
    if not rid:
        return None

    # 1) Registry
    reg = _load_registry()
    forms = reg.get("forms") or []
    for row in forms:
        if (row or {}).get("form_id") == rid:
            rel = (row or {}).get("path") or ""
            if rel:
                p = (config.BASE_DIR / rel).resolve()
                return p if p.exists() else None

    # 2) Bucket scan
    for b in VALID_BUCKETS:
        p = config.EXPL_DIR / b / f"{rid}.json"
        if p.exists():
            return p
    return None

def _load_canonical_explainer(canonical_id: str) -> Optional[Dict[str, Any]]:
    p = _find_explainer_path(canonical_id)
    if not p:
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f) or None
    except Exception:
        return None

# ---- Strict LLM generation (embedded here per your request) ----

def _build_messages_for_strict_explainer(
    *, canonical_id: str, bucket_guess: str, title_guess: str, text_snippet: str
) -> List[Dict[str, str]]:
    schema_rules = config.EXPLAINER_SCHEMA_NOTES.strip() if hasattr(config, "EXPLAINER_SCHEMA_NOTES") else """
Return a JSON with keys: title, form_id, sections[{title,fields[{label,summary}]}],
metrics{tp,fp,fn,precision,recall,f1}, canonical_id, bucket, schema_version, created_at, updated_at, aliases[].
"""
    sys = (
        "You are IntelliForm Explainer. Output strictly ONE JSON object. "
        "NO markdown, no commentary. Use ONLY information explicitly present in the provided text excerpt. "
        "Every field label must be a substring of the provided text (case-insensitive, whitespace-normalized). "
        "If a label cannot be verified, omit it. Prefer exact printed wording; trim trailing colons."
    )
    user = f"""
Generate the explainer JSON for a PDF form.

Canonical template hash (ID): {canonical_id}
Bucket guess: {bucket_guess}
Title guess: {title_guess}

Use ONLY what can be justified from this text (OCR excerpt):
---
{text_snippet}
---

Schema & strict rules:
{schema_rules}

Output ONLY the JSON.
""".strip()
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

def _chat_completion_strict_json(model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
    """
    Uses config.chat_completion (already vendor-neutral) with JSON response_format.
    """
    return config.chat_completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        enforce_json=True,
    )

def _generate_explainer_json_strict_local(
    *,
    canonical_id: str,
    bucket_guess: str,
    title_guess: str,
    text_snippet: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Returns a parsed dict (never writes to disk). Provides a minimal scaffold if LLM fails.
    """
    m = model or config.ENGINE_MODEL
    mt = max_tokens or config.MAX_TOKENS
    tmp = config.TEMPERATURE if temperature is None else float(temperature)

    messages = _build_messages_for_strict_explainer(
        canonical_id=canonical_id,
        bucket_guess=bucket_guess,
        title_guess=title_guess,
        text_snippet=text_snippet or "",
    )
    raw = _chat_completion_strict_json(m, messages, mt, tmp)

    def _parse(s: str) -> Dict[str, Any]:
        try:
            return json.loads((s or "").strip())
        except Exception:
            return {}

    data = _parse(raw)
    if not data or not isinstance(data, dict):
        # minimal scaffold
        return {
            "title": title_guess,
            "form_id": config.sanitize_form_id(title_guess or canonical_id),
            "sections": [],
            "metrics": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "canonical_id": canonical_id,
            "bucket": bucket_guess,
            "schema_version": 1,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "aliases": [],
        }

    # Canonical stamps / defaults
    data.setdefault("title", title_guess or canonical_id)
    data.setdefault("form_id", config.sanitize_form_id(title_guess or canonical_id))
    data["canonical_id"] = canonical_id
    data["bucket"] = bucket_guess
    try:
        data["schema_version"] = int(data.get("schema_version") or 1)
    except Exception:
        data["schema_version"] = 1
    data.setdefault("created_at", _now_iso())
    data["updated_at"] = _now_iso()
    data.setdefault("aliases", [])
    return data

# ---- Local degrading (programmatic, deterministic) ----

@dataclass
class DegradeParams:
    drop_rate: float
    mislabel_rate: float
    vague_rate: float
    seed: Optional[int]

_GENERIC_LABELS = [
    "Name", "Contact", "Details", "Information", "Address", "Date", "Place", "Reference", "ID", "Notes",
]

_VAGUE_SUMMARIES = [
    "Provide details.", "Enter information as needed.", "Write the appropriate value.",
    "Fill this section.", "Complete as applicable.", "Enter info.", "Provide required info.",
]

def _truncate_label(label: str) -> str:
    # mild degradation: shorten or generalize
    s = (label or "").strip()
    if not s:
        return s
    # remove trailing colon and extra words at the end
    s = s.rstrip(":").strip()
    parts = s.split()
    if len(parts) >= 2:
        return " ".join(parts[:-1])
    return parts[0] if parts else s

def _mislabel(label: str, rng: random.Random) -> str:
    if rng.random() < 0.5:
        return _truncate_label(label)
    return rng.choice(_GENERIC_LABELS)

def _vague(summary: str, rng: random.Random) -> str:
    s = (summary or "").strip()
    if not s:
        return rng.choice(_VAGUE_SUMMARIES)
    # shorten to first sentence-ish or replace
    if rng.random() < 0.6:
        return rng.choice(_VAGUE_SUMMARIES)
    return (s.split(".")[0] + ".").strip()

def _degrade_metrics(payload: Dict[str, Any], rng: random.Random) -> None:
    m = dict(payload.get("metrics") or {})
    # baseline aims to be clearly worse: knock ~15–35% off P/R with some noise
    def _f(x, pct):
        try:
            v = float(x)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, v * (1.0 - pct)))
    noise = 0.15 + 0.20 * rng.random()  # 0.15..0.35
    p = _f(m.get("precision", 0.8), noise)
    r = _f(m.get("recall", 0.8), noise * (0.9 + 0.2 * rng.random()))
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    # estimate tp/fp/fn from counts if present; otherwise keep rough shell
    tp = int(max(0, round((m.get("tp") or 40) * (1 - noise))))
    fn = int(max(0, round((m.get("fn") or 20) * (1 + noise))))
    fp = int(max(0, round((m.get("fp") or 20) * (1 + noise))))
    payload["metrics"] = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(f"{p:.3f}"),
        "recall": float(f"{r:.3f}"),
        "f1": float(f"{f1:.3f}"),
    }

def _degrade_local(payload: Dict[str, Any], p: DegradeParams) -> Dict[str, Any]:
    rng = _rng(p.seed)
    data = _deepcopy(payload)

    sections = data.get("sections") or []
    new_sections: List[Dict[str, Any]] = []
    for sec in sections:
        title = (sec or {}).get("title") or "Section"
        fields = (sec or {}).get("fields") or []
        kept_fields: List[Dict[str, str]] = []
        for f in fields:
            if not isinstance(f, dict):
                continue
            label = str(f.get("label", "") or "")
            summary = str(f.get("summary", "") or "")

            # Drop some fields
            if rng.random() < p.drop_rate:
                continue

            # Mislabel a portion
            if rng.random() < p.mislabel_rate:
                label = _mislabel(label, rng)

            # Vague summaries
            if rng.random() < p.vague_rate:
                summary = _vague(summary, rng)

            kept_fields.append({"label": label, "summary": summary})

        if kept_fields:
            new_sections.append({"title": title, "fields": kept_fields})

    data["sections"] = new_sections
    _degrade_metrics(data, rng)

    # update timestamps but DO NOT write to disk
    data["updated_at"] = _now_iso()
    return data

# ---- LLM-based degrading (non-destructive) ----

def _degrade_llm(payload: Dict[str, Any], p: DegradeParams) -> Dict[str, Any]:
    """
    Ask the model to intentionally underperform: fewer fields, ambiguous labels,
    vague summaries, and lowered metrics. Return JSON only.
    """
    base = _deepcopy(payload)
    # Keep only minimal context in the prompt (schema + “degrade” rules)
    sys = (
        "You are a baseline-quality document explainer that intentionally performs worse than the production system. "
        "You will output ONE JSON object only (no markdown). "
        "Intentionally drop a portion of fields, misname some labels to be ambiguous, "
        "write vague summaries, and decrease the metrics realistically."
    )
    user = f"""
Given this ORIGINAL explainer JSON:

---
{json.dumps(base, ensure_ascii=False)}
---

Produce a DEGRADED explainer JSON with the SAME schema, applying:
- Drop about {int(p.drop_rate*100)}% of fields (random-ish, but keep structure plausible).
- Mislabel about {int(p.mislabel_rate*100)}% of remaining labels (truncate/generalize).
- Make about {int(p.vague_rate*100)}% of summaries vague/unclear (“Provide details.”, etc.).
- Lower precision/recall/F1 by roughly 15–35% (keep tp/fp/fn consistent with the change).
- Keep title, canonical_id, bucket, schema_version. Update updated_at. Do NOT add commentary.

Return ONLY the JSON object, no extra text.
""".strip()

    messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    raw = config.chat_completion(
        model=config.ENGINE_MODEL,
        messages=messages,
        max_tokens=min(1800, config.MAX_TOKENS),
        temperature=0.2,  # keep deterministic-ish
        enforce_json=True,
    )
    try:
        data = json.loads((raw or "").strip())
        if isinstance(data, dict) and (data.get("sections") is not None):
            # Ensure mandatory stamps
            data.setdefault("title", base.get("title"))
            data.setdefault("form_id", base.get("form_id"))
            data["canonical_id"] = base.get("canonical_id")
            data["bucket"] = base.get("bucket")
            data["schema_version"] = int((data.get("schema_version") or base.get("schema_version") or 1))
            data.setdefault("created_at", base.get("created_at") or _now_iso())
            data["updated_at"] = _now_iso()
            return data
    except Exception:
        pass

    # Fallback to local degrade if LLM path fails
    return _degrade_local(base, p)

# ---- Public entrypoint ----

def get_degraded_explainer(
    canonical_id: str,
    *,
    bucket_hint: Optional[str] = None,
    title_hint: Optional[str] = None,
    pdf_disk_path: Optional[str] = None,  # used only when no canonical explainer exists
    backend: Optional[str] = None,        # "llm" | "local"
    seed: Optional[int] = None,
    drop_rate: Optional[float] = None,
    mislabel_rate: Optional[float] = None,
    vague_rate: Optional[float] = None,
    strict_ephemeral: bool = True,        # keep True to guarantee no writes
) -> Dict[str, Any]:
    """
    Returns an in-memory degraded explainer dict. Never mutates or writes the canonical.
    Behavior:
      - If a saved explainer exists → load and degrade.
      - Else → generate strict explainer via LLM from snippet (if possible), then degrade.
      - Never save the degraded result (strict_ephemeral=True).
    """
    # Normalize knobs from config or params
    mode_backend = (backend or getattr(config, "BASELINE_BACKEND", "llm")).strip().lower()
    pr_drop  = _norm_rate(drop_rate,  getattr(config, "BASELINE_DROP_RATE", 0.35))
    pr_mis   = _norm_rate(mislabel_rate, getattr(config, "BASELINE_MISLABEL_RATE", 0.15))
    pr_vague = _norm_rate(vague_rate, getattr(config, "BASELINE_VAGUE_RATE", 0.60))
    the_seed = seed if seed is not None else getattr(config, "BASELINE_SEED", None)
    params = DegradeParams(drop_rate=pr_drop, mislabel_rate=pr_mis, vague_rate=pr_vague, seed=the_seed)

    # 1) Load canonical if present
    canonical = _load_canonical_explainer(canonical_id)

    # 2) Else create a baseline-quality explainer using strict LLM context
    if canonical is None:
        bucket_guess = (bucket_hint or "government").lower()
        if bucket_guess not in VALID_BUCKETS:
            bucket_guess = "government"
        title_guess = title_hint or canonical_id

        snippet = ""
        if pdf_disk_path and os.path.exists(pdf_disk_path):
            try:
                snippet = config.quick_text_snippet(pdf_disk_path, max_chars=6000)
            except Exception:
                snippet = ""

        canonical = _generate_explainer_json_strict_local(
            canonical_id=canonical_id,
            bucket_guess=bucket_guess,
            title_guess=title_guess,
            text_snippet=snippet or "",
        )

    # 3) Degrade (LLM or local)
    if mode_backend == "llm":
        degraded = _degrade_llm(canonical, params)
    else:
        degraded = _degrade_local(canonical, params)

    # 4) Ensure strictly ephemeral behavior: no files written here
    # (If ever needed, an API layer may choose to cache to /out/baseline/, but the default is no write.)
    return degraded
