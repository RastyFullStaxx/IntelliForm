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
    - bucket and aliases are optional.
    - Aliases are merged case-insensitively with any existing aliases.
    """
    reg = load_registry(path)
    forms: List[Dict[str, Any]] = reg.get("forms", [])

    idx = next((i for i, f in enumerate(forms) if f.get("form_id") == form_id), None)

    # normalize path slashes
    rel_path = rel_path.replace("\\", "/")

    # normalize aliases: strip/keep non-empty, case-fold for dedupe but preserve original casing of first occurrence
    def _norm_aliases(vals: Optional[List[str]]):
        raw = [a.strip() for a in (vals or []) if a and a.strip()]
        seen: Dict[str, str] = {}
        for a in raw:
            key = a.casefold()
            if key not in seen:
                seen[key] = a  # preserve first-seen casing
        return list(seen.values()), seen

    new_aliases_list, new_aliases_map = _norm_aliases(aliases)

    new_entry = {
        "form_id": form_id,
        "title": title or form_id,
        "path": rel_path,
    }
    if bucket:
        new_entry["bucket"] = bucket
    if new_aliases_list:
        new_entry["aliases"] = new_aliases_list

    if idx is None:
        forms.append(new_entry)
    else:
        existing = forms[idx]
        merged = {**existing, **new_entry}  # path/title/bucket from new_entry override

        # Merge aliases case-insensitively
        old_aliases_list, old_aliases_map = _norm_aliases(existing.get("aliases", []))
        if old_aliases_list or new_aliases_list:
            merged_map = {**old_aliases_map}
            for k, v in new_aliases_map.items():
                merged_map.setdefault(k, v)
            merged["aliases"] = sorted(list(merged_map.values()))
        forms[idx] = merged

    reg["forms"] = forms
    save_registry(path, reg)

def find_by_hash(path: str, form_id: str) -> Optional[Dict[str, Any]]:
    reg = load_registry(path)
    for f in reg.get("forms", []):
        if f.get("form_id") == form_id:
            return f
    return None

def find_by_alias(path: str, alias: str) -> Optional[Dict[str, Any]]:
    """
    Return the registry entry whose aliases contain `alias` (case-insensitive).
    """
    q = (alias or "").strip()
    if not q:
        return None
    qkey = q.casefold()
    reg = load_registry(path)
    for f in reg.get("forms", []):
        for a in (f.get("aliases") or []):
            try:
                if a and a.casefold() == qkey:
                    return f
            except Exception:
                continue
    return None

def find_by_any(path: str, key: str) -> Optional[Dict[str, Any]]:
    """
    Resolve either by canonical hash (form_id) or by alias, case-insensitively.
    """
    if not key:
        return None
    hit = find_by_hash(path, key)
    if hit:
        return hit
    return find_by_alias(path, key)
