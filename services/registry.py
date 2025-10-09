# services/registry.py
import os, json
from typing import Dict, Any, List, Optional

def _ensure_dir(p: str) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)

def load_registry(path: str) -> Dict[str, Any]:
    """
    Registry schema (hash-first):
      {
        "forms": [
          {
            "form_id": "<HASH>",
            "title": "Pretty Title",
            "path": "explanations/<bucket>/<HASH>.json",
            "bucket": "healthcare|banking|government|tax",
            "aliases": ["AXA_MotorClaimForm", "original_filename_stem", ...]
          },
          ...
        ]
      }
    """
    if not os.path.exists(path):
        return {"forms": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            data = json.loads(txt) if txt else {}
            if not isinstance(data, dict):
                return {"forms": []}
            data.setdefault("forms", [])
            return data
    except Exception:
        return {"forms": []}

def save_registry(path: str, data: Dict[str, Any]) -> None:
    _ensure_dir(path)
    if not isinstance(data, dict):
        data = {"forms": []}
    data.setdefault("forms", [])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def upsert_registry(path: str, form_id: str, *, title: str, rel_path: str,
                    bucket: Optional[str] = None, aliases: Optional[List[str]] = None) -> None:
    """
    Insert or replace an entry keyed by form_id (the canonical HASH).
    - rel_path should be relative to repo root (e.g., explanations/healthcare/<HASH>.json)
    - bucket and aliases are optional but recommended.
    - merges aliases if an entry already exists.
    """
    reg = load_registry(path)
    forms: List[Dict[str, Any]] = reg.get("forms", [])

    idx = next((i for i, f in enumerate(forms) if f.get("form_id") == form_id), None)
    new_entry = {
        "form_id": form_id,
        "title": title or form_id,
        "path": rel_path.replace("\\", "/"),
    }
    if bucket:
        new_entry["bucket"] = bucket
    if aliases:
        new_entry["aliases"] = sorted(list({a for a in aliases if a}))

    if idx is None:
        forms.append(new_entry)
    else:
        # merge with existing
        existing = forms[idx]
        merged = {
            **existing,
            **new_entry,  # new values override existing (path/title/bucket)
        }
        # merge aliases sets if both have them
        a_old = set(existing.get("aliases", []) or [])
        a_new = set(new_entry.get("aliases", []) or [])
        if a_old or a_new:
            merged["aliases"] = sorted(list(a_old | a_new))
        forms[idx] = merged

    reg["forms"] = forms
    save_registry(path, reg)

def find_by_hash(path: str, form_id: str) -> Optional[Dict[str, Any]]:
    reg = load_registry(path)
    for f in reg.get("forms", []):
        if f.get("form_id") == form_id:
            return f
    return None
