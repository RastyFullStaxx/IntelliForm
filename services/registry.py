# services/registry.py
import os, json
from typing import Dict

def load_registry(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            return json.loads(txt) if txt else {}
    except Exception:
        return {}

def save_registry(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def upsert_registry(path: str, form_id: str, *, title: str, rel_path: str) -> None:
    reg = load_registry(path)
    reg[form_id] = {"title": title, "path": rel_path.replace("\\","/")}
    save_registry(path, reg)

