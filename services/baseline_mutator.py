# services/baseline_mutator.py
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts import config


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
  rid = (canonical_id or "").strip()
  if not rid:
    return None

  reg = _load_registry()
  forms = reg.get("forms") or []
  for row in forms:
    if (row or {}).get("form_id") == rid:
      rel = (row or {}).get("path") or ""
      if rel:
        p = (config.BASE_DIR / rel).resolve()
        return p if p.exists() else None

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
  m = model or config.ENGINE_MODEL
  mt = max_tokens or config.MAX_TOKENS
  tmp = config.TEMPERATURE if temperature is None else float(temperature)

  messages = _build_messages_for_strict_explainer(
    canonical_id=canonical_id, bucket_guess=bucket_guess, title_guess=title_guess, text_snippet=text_snippet or ""
  )
  raw = _chat_completion_strict_json(m, messages, mt, tmp)

  def _parse(s: str) -> Dict[str, Any]:
    try:
      return json.loads((s or "").strip())
    except Exception:
      return {}

  data = _parse(raw)
  if not data or not isinstance(data, dict):
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


@dataclass
class DegradeParams:
  drop_rate: float
  mislabel_rate: float
  vague_rate: float
  seed: Optional[int]


def _degrade_sections(sections: List[Dict[str, Any]], params: DegradeParams) -> List[Dict[str, Any]]:
  rng = _rng(params.seed)
  out = []
  for sec in sections:
    fields = []
    for f in sec.get("fields", []):
      f = _deepcopy(f)
      if rng.random() < params.drop_rate:
        continue
      if rng.random() < params.mislabel_rate:
        f["label"] = f.get("label", "")[::-1][:15] or "—"
      if rng.random() < params.vague_rate:
        f["summary"] = "N/A"
      fields.append(f)
    sec2 = _deepcopy(sec)
    sec2["fields"] = fields
    out.append(sec2)
  return out


def get_degraded_explainer(canonical_id: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
  if not config.BASELINE_MODE:
    return None

  base = _load_canonical_explainer(canonical_id)
  if not base:
    return None

  p = DegradeParams(
    drop_rate=_norm_rate((params or {}).get("drop_rate"), config.BASELINE_DROP_RATE),
    mislabel_rate=_norm_rate((params or {}).get("mislabel_rate"), config.BASELINE_MISLABEL_RATE),
    vague_rate=_norm_rate((params or {}).get("vague_rate"), config.BASELINE_VAGUE_RATE),
    seed=(params or {}).get("seed", config.BASELINE_SEED),
  )

  degraded = _deepcopy(base)
  degraded["sections"] = _degrade_sections(base.get("sections") or [], p)
  degraded["degraded_from"] = canonical_id
  degraded["baseline_params"] = p.__dict__
  return degraded
