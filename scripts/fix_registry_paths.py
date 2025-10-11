# scripts/fix_registry_paths.py
from pathlib import Path
import json, os

BASE_DIR = Path(__file__).resolve().parent.parent
REGISTRY = BASE_DIR / "explanations" / "registry.json"

def relativize(p: str) -> str:
    if not p:
        return p
    p = p.replace("\\", "/")
    # If already relative (starts with 'explanations/'), keep it
    if p.startswith("explanations/"):
        return p
    # If absolute, turn into BASE_DIR-relative
    try:
        rel = os.path.relpath(p, BASE_DIR)
    except Exception:
        rel = p
    return rel.replace("\\", "/").lstrip("/")

def main():
    if not REGISTRY.exists():
        print(f"Registry not found: {REGISTRY}")
        return
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    changed = False
    for f in data.get("forms", []):
        old = f.get("path", "")
        new = relativize(old)
        if new != old:
            f["path"] = new
            changed = True
            print(f"Fixed: {old}  ->  {new}")
    if changed:
        REGISTRY.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Registry paths normalized.")
    else:
        print("No changes needed.")

if __name__ == "__main__":
    main()
