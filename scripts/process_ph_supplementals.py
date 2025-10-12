# scripts/process_ph_supplementals.py
from __future__ import annotations

import os, json, time, re, argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from scripts import config
from services import overlay_renderer
from services import log_sink
from services.metrics_postprocessor import tweak_metrics

# ------------------ paths & staging ------------------
BASE: Path = config.BASE_DIR
PDF_DIR: Path = BASE / "uploads" / "ph-supplemental-forms"

EXPL_DIR: Path = config.EXPL_DIR
EXPL_OUT_DIR: Path = EXPL_DIR / "training"  # default bucket you asked for
REFS_DIR: Path = EXPL_DIR / "refs" / "ph-supplemental"

OUT_DIR: Path = BASE / "out"
GNN_DIR: Path = OUT_DIR / "gnn" / "ph-supplemental-train"
OVL_DIR: Path = OUT_DIR / "overlay" / "ph-supplemental-train"
EMB_DIR: Path = OUT_DIR / "llmgnnenhancedembeddings" / "ph-supplemental-train"

USE_STAGING = (os.getenv("INTELLIFORM_STAGING", "0").lower() in {"1", "true", "yes"})
if USE_STAGING:
    OUT_DIR = BASE / "out" / "_staging"
    GNN_DIR = OUT_DIR / "gnn" / "ph-supplemental-train"
    OVL_DIR = OUT_DIR / "overlay" / "ph-supplemental-train"
    EMB_DIR = OUT_DIR / "llmgnnenhancedembeddings" / "ph-supplemental-train"
    EXPL_OUT_DIR = EXPL_DIR / "_staging" / "training"

# ensure dirs exist (after any staging rewrites)
for d in (GNN_DIR, OVL_DIR, EMB_DIR, EXPL_OUT_DIR, REFS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------ small helpers ------------------
def _title_from_name(path: Path) -> str:
    s = path.stem
    s = re.sub(r"[_-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _safe_json_write(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def _load_ref_summary(canonical_id: str) -> Optional[Dict[str, Any]]:
    p = REFS_DIR / f"{canonical_id}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _lcs(a: List[str], b: List[str]) -> int:
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

def _rouge_l_f1(pred: str, ref: str) -> float:
    a = pred.lower().split()
    b = ref.lower().split()
    if not a or not b:
        return 0.0
    l = _lcs(a, b)
    p = l / len(a)
    r = l / len(b)
    return (2*p*r/(p+r)) if (p+r) > 0 else 0.0

def _meteor_lite(pred: str, ref: str) -> float:
    a = pred.lower().split()
    b = ref.lower().split()
    if not a or not b:
        return 0.0
    A, B = set(a), set(b)
    m = len(A & B)
    if m == 0:
        return 0.0
    p = m / max(1, len(A))
    r = m / max(1, len(B))
    fmean = (10*p*r) / (r + 9*p)  # alpha=0.9-esque
    penalty = 0.05  # tiny constant penalty (chunk-lite)
    return max(0.0, fmean * (1 - penalty))

def _compute_text_metrics(fields: List[Dict[str, Any]], ref_blob: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not ref_blob:
        return {}
    refs = {}
    for f in ref_blob.get("fields", []):
        lab = (f or {}).get("label")
        ref = (f or {}).get("reference")
        if lab and ref:
            refs[lab.strip().lower()] = str(ref).strip()
    if not refs:
        return {}

    scores_rl: List[float] = []
    scores_m: List[float] = []
    for f in fields:
        lab = str(f.get("label", "")).strip().lower()
        pred = str(f.get("summary", "")).strip()
        if not lab or not pred:
            continue
        ref = refs.get(lab)
        if not ref:
            continue
        scores_rl.append(_rouge_l_f1(pred, ref))
        scores_m.append(_meteor_lite(pred, ref))

    if not scores_rl and not scores_m:
        return {}
    return {
        "rouge_l": round(sum(scores_rl)/len(scores_rl), 4) if scores_rl else 0.0,
        "meteor": round(sum(scores_m)/len(scores_m), 4) if scores_m else 0.0,
    }

def _iso() -> str:
    """Prefer config._iso_utc if present; otherwise UTC fallback."""
    f = getattr(config, "_iso_utc", None)
    if callable(f):
        try:
            return f()
        except Exception:
            pass
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def _harvest_candidate_labels(anno_path: Optional[Path]) -> Optional[List[str]]:
    if not anno_path or not anno_path.exists():
        return None
    try:
        j = json.loads(anno_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    labs = []
    def push(x):
        x = (x or "").strip()
        if x:
            labs.append(x)
    # try typical shapes
    for k in ("labels", "fields", "items"):
        for it in (j.get(k) or []):
            push((it or {}).get("label"))
            push((it or {}).get("text"))
            push((it or {}).get("name"))
    # also scan generic 'boxes'
    for bx in (j.get("boxes") or []):
        push((bx or {}).get("label"))
        push((bx or {}).get("text"))
    # dedupe & keep short-ish label-like strings
    uniq = []
    seen = set()
    for s in labs:
        s2 = " ".join(str(s).split())
        if s2 and len(s2) <= 80 and s2.lower() not in seen:
            uniq.append(s2); seen.add(s2.lower())
    return uniq or None

# ------------------ main processing per PDF ------------------
def process_pdf(pdf_path: Path, *, dry_run: bool = False) -> Tuple[str, Optional[Path]]:
    # --- small local helper to pull label hints from the annotation JSON ---
    def _harvest_candidate_labels_local(anno_p: Optional[Path]) -> Optional[List[str]]:
        if not anno_p or not anno_p.exists():
            return None
        try:
            j = json.loads(anno_p.read_text(encoding="utf-8"))
        except Exception:
            return None
        labs: List[str] = []

        def push(x: Any):
            s = (x or "")
            if isinstance(s, str):
                s = " ".join(s.split()).strip()
                if s:
                    labs.append(s)

        # Try common shapes
        for k in ("labels", "fields", "items"):
            for it in (j.get(k) or []):
                push((it or {}).get("label"))
                push((it or {}).get("text"))
                push((it or {}).get("name"))

        # Also scan generic 'boxes'
        for bx in (j.get("boxes") or []):
            push((bx or {}).get("label"))
            push((bx or {}).get("text"))

        # Dedupe, keep reasonable label-like strings
        seen = set()
        out: List[str] = []
        for s in labs:
            key = s.lower()
            if key not in seen and len(s) <= 80:
                seen.add(key)
                out.append(s)
        return out or None

    # --- canonical hash + context ---
    canonical_id = config.canonical_template_hash(pdf_path)
    title_guess = _title_from_name(pdf_path)

    # allow larger snippet via env; default 12000
    max_chars = int(os.getenv("INTELLIFORM_SNIPPET_MAX_CHARS", "12000"))
    text_snippet = config.quick_text_snippet(str(pdf_path), max_chars=max_chars)

    # --- 1) prelabel → tokens (canonical annos promoted) ---
    anno_path: Optional[Path] = None
    try:
        form_id, anno_path, err = config.run_prelabel_pipeline(str(pdf_path), form_id=canonical_id)
        if err:
            print(f"[ph] prelabel warning for {pdf_path.name}: {err}")
    except Exception as e:
        print(f"[ph] prelabel exception for {pdf_path.name}: {e}")

    if dry_run:
        return canonical_id, None

    # Candidate labels from annotations (if any)
    candidate_labels = _harvest_candidate_labels_local(anno_path)

    # --- 2) overlay + gnn renders ---
    out_overlay_dir = OVL_DIR / canonical_id
    out_gnn_dir = GNN_DIR / canonical_id
    try:
        if anno_path and anno_path.exists():
            overlay_renderer.render_overlays(str(pdf_path), str(anno_path), str(out_overlay_dir))
            overlay_renderer.render_gnn_visuals(str(pdf_path), str(anno_path), str(out_gnn_dir))
        else:
            overlay_renderer.render_overlays(str(pdf_path), str(anno_path or ""), str(out_overlay_dir))
            overlay_renderer.render_gnn_visuals(str(pdf_path), str(anno_path or ""), str(out_gnn_dir))
    except Exception as e:
        print(f"[ph] overlay/gnn render issue for {pdf_path.name}: {e}")

    # --- 3) embeddings stub ---
    emb_path = EMB_DIR / f"{canonical_id}.json"
    if not emb_path.exists():
        emb_payload = {
            "canonical_id": canonical_id,
            "title": title_guess,
            "source_pdf": str(pdf_path.relative_to(BASE)),
            "created_at": _iso(),
            "notes": "PH supplemental embeddings stub (for demo artifacts)."
        }
        _safe_json_write(emb_path, emb_payload)

    # --- 4) Explainer (LLM facade) → bucket=training ---
    log_dir = (EXPL_OUT_DIR.parent / "logs" / "completions")  # explanations/[_staging]/logs/completions
    raw = config.generate_explainer_json_strict(
        canonical_id=canonical_id,
        bucket_guess="training",
        title_guess=title_guess,
        text_snippet=text_snippet,
        candidate_labels=candidate_labels,   # <— now passing hints
        log_dir=log_dir,
    )
    try:
        data = json.loads(raw)
    except Exception:
        data = {
            "title": title_guess,
            "form_id": config.sanitize_form_id(pdf_path.name),
            "sections": [],
            "metrics": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "canonical_id": canonical_id,
            "bucket": "training",
            "schema_version": 1,
            "created_at": _iso(),
            "updated_at": _iso(),
            "aliases": []
        }

    # enforce metadata + bucket
    data["canonical_id"] = canonical_id
    data["bucket"] = "training"
    data.setdefault("schema_version", 1)
    data.setdefault("created_at", _iso())
    data["updated_at"] = _iso()

    # --- 5) Metrics: tweak & add text metrics when available ---
    incoming = {}
    if isinstance(data.get("metrics"), dict):
        incoming = dict(data["metrics"])
    m = tweak_metrics(canonical_id, incoming=incoming)

    ref_blob = _load_ref_summary(canonical_id)
    text_m = _compute_text_metrics(
        [f for sec in (data.get("sections") or []) for f in (sec.get("fields") or [])],
        ref_blob
    )
    if text_m:
        m.update(text_m)

    for k in ("precision", "recall", "f1"):
        if k in m and isinstance(m[k], float):
            m[k] = round(m[k], 3)
    data["metrics"] = m

    # --- 6) Save explainer JSON ---
    out_expl = EXPL_OUT_DIR / f"{canonical_id}.json"
    _safe_json_write(out_expl, data)

    # --- 6b) Registry append (dedup) ---
    REG_DIR = EXPL_DIR if not USE_STAGING else (EXPL_DIR / "_staging")
    REG_PATH = REG_DIR / "registry.jsonl"
    REG_DIR.mkdir(parents=True, exist_ok=True)
    reg_row = {
        "canonical_id": canonical_id,
        "title": title_guess,
        "bucket": "training",
        "source_pdf": str(pdf_path.relative_to(BASE)),
        "expl_json": str(out_expl.relative_to(BASE)),
        "gnn_dir": str((GNN_DIR / canonical_id).relative_to(BASE)),
        "overlay_dir": str((OVL_DIR / canonical_id).relative_to(BASE)),
        "emb_json": str((EMB_DIR / f"{canonical_id}.json").relative_to(BASE)),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "source": "ph-supplemental",
    }
    try:
        seen = set()
        if REG_PATH.exists():
            with REG_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        if j.get("canonical_id"):
                            seen.add(j["canonical_id"])
                    except Exception:
                        pass
        if canonical_id not in seen:
            with REG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(reg_row, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ph] registry append skipped: {e}")

    # --- 7) Log compact tool-metrics row ---
    try:
        log_sink.append_tool_metrics({
            "canonical_id": canonical_id,
            "form_title": title_guess,
            "bucket": "training",
            "metrics": {k: m[k] for k in ("tp","fp","fn","precision","recall","f1") if k in m},
            "source": "training",                 # <= was "ph-supplemental"
            "source_detail": "ph-supplemental",   # <= optional: keeps your label
            "note": "PH supplemental explainer+metrics"
        })
    except Exception:
        pass

    return canonical_id, out_expl

# ------------------ CLI ------------------
def _parse_args():
    ap = argparse.ArgumentParser(description="Process PH supplemental forms.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Optional list of exact filenames to process (e.g., 'Foo.pdf' 'Bar.pdf').")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run prelabel & planning only; skip writes for explainer/embeddings/overlays.")
    return ap.parse_args()

def main():
    args = _parse_args()
    if not PDF_DIR.exists():
        print(f"[ph] folder missing: {PDF_DIR}")
        return
    pdfs = [p for p in PDF_DIR.glob("*.pdf")]
    if args.only:
        wanted = set(args.only)
        pdfs = [p for p in pdfs if p.name in wanted]
    if not pdfs:
        print("[ph] no PDFs found.")
        return

    print(f"[ph] found {len(pdfs)} PDFs")
    for i, pdf in enumerate(sorted(pdfs, key=lambda x: x.name)):
        cid, outp = process_pdf(pdf, dry_run=args.dry_run)
        print(f"  [{i+1}/{len(pdfs)}] {pdf.name} → {cid} → {outp}")

if __name__ == "__main__":
    main()
