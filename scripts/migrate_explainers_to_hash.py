#!/usr/bin/env python3
"""
One-shot migration to TEMPLATE-based canonical IDs + registry upgrade.

- Prefers PyMuPDF render (annots=False) + perceptual hashes per page -> SHA256(joined)
- Fallback: pypdf strip AcroForm /V and /AP -> normalized text -> SHA256
- Final fallback: SHA256(file bytes)

Writes new explainers to: explanations/<bucket>/<HASH>.json
Upserts explanations/registry.json to the new {"forms":[...]} schema keyed by <HASH>,
carrying over titles from your current map-shaped registry (if present).
"""

import io, re, json, hashlib
from datetime import datetime
from pathlib import Path

# ---------------- CONFIG (edit paths if different) ----------------
ROOT = Path(".").resolve()
UPLOADS = ROOT / "uploads"
EXPLAINERS = ROOT / "explanations"
REGISTRY = EXPLAINERS / "registry.json"

MAPPINGS = [
    # --- BANKING ---
    {"old_id":"BDO_RealEstate_OfferToBuy_Individual", "bucket":"banking", "pdf":"uploads/Banking_RealEstateIndividualBDO.pdf"},
    {"old_id":"BDO_TenantsInformationSheet",           "bucket":"banking", "pdf":"uploads/Banking_UXD_PMUTenantsInfoSheet.pdf"},
    {"old_id":"FAMI_AccountOpeningForm_Individual",    "bucket":"banking", "pdf":"uploads/Finance_FAMIAccountOpeningForm.pdf"},
    {"old_id":"FAMI_AccountOpeningForm",               "bucket":"banking", "pdf":"uploads/Finance_FAMIAccountOpeningForm.pdf"},
    {"old_id":"Metrobank_AccountInformationAndEnrollment","bucket":"banking","pdf":"uploads/Finance_AccountInformationFormAndEnrollmentToElectronicPlatformsWithStandardSettlementInstruction.pdf"},
    {"old_id":"SLAMCI_InvestorInfoForm2024",           "bucket":"banking", "pdf":"uploads/Finance_SLAMCIInvestorInfoForm2024FCPA.pdf"},

    # --- GOVERNMENT ---
    {"old_id":"COMELEC_CEF1_VoterRegistration",        "bucket":"government", "pdf":"uploads/Goverment_COMELECVoterRegistrationApplicationForm.pdf"},
    {"old_id":"LTO_DriverLicenseApplication",          "bucket":"government", "pdf":"uploads/Government_DriverLicenseApplicationForm.pdf"},
    {"old_id":"PagIBIG_MembershipRegistration",        "bucket":"government", "pdf":"uploads/Government_PagIBIGFundMembershipRegistration.pdf"},
    {"old_id":"PhilHealth_MemberRegistration",         "bucket":"government", "pdf":"uploads/Government_PhilHealthMemberRegistrationForm.pdf"},
    {"old_id":"PSA_BirthCertificateRequest",           "bucket":"government", "pdf":"uploads/Government_BirthCertificateRequestForm.pdf"},

    # --- HEALTHCARE ---
    {"old_id":"Allianz_IHP_ClaimForm",                 "bucket":"healthcare", "pdf":"uploads/Healthcare_Allianz_IHP_ClaimForm.pdf"},
    {"old_id":"AXA_MotorClaimForm",                    "bucket":"healthcare", "pdf":"uploads/Healthcare_AXA_MotorClaimForm.pdf"},
    {"old_id":"FWD_TPD_DismembermentClaimForm",        "bucket":"healthcare", "pdf":"uploads/Healthcare_FWD_TPD_DismembermentClaimForm.pdf"},
    {"old_id":"Manulife_APS_AmyotrophicLateralSclerosis","bucket":"healthcare", "pdf":"uploads/Healthcare_Manulife_APS_ALS.pdf"},
    {"old_id":"SunLife_AccountOpeningForm",            "bucket":"healthcare", "pdf":"uploads/Healthcare_SunLife_AccountOpeningForm.pdf"},

    # --- TAX ---
    {"old_id":"BIR_AnnualReturn1604E",                 "bucket":"tax", "pdf":"uploads/Tax_BIRAnnualInformationReturn1604E.pdf"},
    {"old_id":"BIR_ApplicationForRefund1913",          "bucket":"tax", "pdf":"uploads/Tax_BIRApplicationForRefund1913.pdf"},
    {"old_id":"BIR_ApplicationForRegistration1901",    "bucket":"tax", "pdf":"uploads/Tax_BIRApplicationForRegistration1901.pdf"},
    {"old_id":"BIR_ApplicationForRegistration1902",    "bucket":"tax", "pdf":"uploads/Tax_BIRApplicationForRegistration1902.pdf"},
    {"old_id":"BIR_PercentageTaxReturn2552",           "bucket":"tax", "pdf":"uploads/Tax_BIRPercentagetaxReturn2552.pdf"},
]
# ------------------------------------------------------------------

_WS = re.compile(r"\s+")
def _norm_text(s: str) -> str:
    return _WS.sub(" ", (s or "").lower()).strip()

def sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def template_hash_via_render(pdf_path: Path) -> str | None:
    """Try PyMuPDF (fitz) + imagehash → phash per page with annots=False."""
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import imagehash
        per_page = []
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(annots=False, alpha=False)  # ignore widgets/annots
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            per_page.append(str(imagehash.phash(img)))
        doc.close()
        joined = "|".join(per_page)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()
    except Exception:
        return None

def template_hash_via_pypdf_text(pdf_path: Path) -> str | None:
    """Strip AcroForm values (/V) & appearance (/AP) using pypdf; hash normalized text."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        root = reader.trailer.get("/Root", {})
        acro = root.get("/AcroForm", {})
        fields = acro.get("/Fields", [])
        for f in fields:
            try:
                if "/V" in f: f["/V"] = ""
                if "/AP" in f: del f["/AP"]
            except Exception:
                continue
        text_parts = []
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                text_parts.append("")
        norm = _norm_text(" ".join(text_parts))
        if not norm:
            return None
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()
    except Exception:
        return None

def canonical_template_hash(pdf_path: Path) -> str:
    # Prefer render-based template hash -> text-based -> bytes
    return (
        template_hash_via_render(pdf_path)
        or template_hash_via_pypdf_text(pdf_path)
        or sha256_bytes(pdf_path)
    )

def load_json(path: Path) -> dict:
    """
    Read JSON robustly:
    - try utf-8
    - fallback to utf-8-sig (strips BOM)
    - final fallback: strip a leading BOM char manually
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Retry with utf-8-sig to strip BOM
        with open(path, "r", encoding="utf-8-sig") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # Last resort: strip a leading BOM if present and parse again
                f.seek(0)
                text = f.read()
                text = text.lstrip("\ufeff")
                return json.loads(text)

def save_json(path: Path, obj: dict):
    """
    Write JSON as UTF-8 (no BOM).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# --- registry helpers that ALSO upgrade your old map-shaped registry ---
def _load_legacy_registry_map() -> dict:
    if not REGISTRY.exists():
        return {}
    try:
        data = load_json(REGISTRY)
    except Exception:
        return {}
    # if it's already {"forms":[...]}, return empty (we'll append during upserts)
    if isinstance(data, dict) and "forms" in data:
        return {}
    # else assume your legacy map shape: { "Key": { "title": "...", "form_id": "..." }, ... }
    return data if isinstance(data, dict) else {}

def _load_registry_forms() -> dict:
    if not REGISTRY.exists():
        return {"forms":[]}
    try:
        data = load_json(REGISTRY)
    except Exception:
        return {"forms":[]}
    if isinstance(data, dict) and "forms" in data and isinstance(data["forms"], list):
        return data
    # legacy: convert map to list with no hashes (we’ll fill later)
    return {"forms":[]}

def upsert_registry(form_id_hash: str, title: str, bucket: str, path: Path, aliases: list[str]):
    reg = _load_registry_forms()
    forms = reg.get("forms", [])
    idx = next((i for i, f in enumerate(forms) if f.get("form_id")==form_id_hash), None)
    entry = {
        "form_id": form_id_hash,
        "title": title or form_id_hash,
        "bucket": bucket,
        "path": str(path).replace("\\","/"),
        "aliases": sorted(list(dict.fromkeys([a for a in aliases if a]))),
    }
    if idx is None:
        forms.append(entry)
    else:
        forms[idx] = entry
    reg["forms"] = forms
    save_json(REGISTRY, reg)

def main():
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    errors = []

    # Load legacy titles so we can preserve them
    legacy_map = _load_legacy_registry_map()  # { old_key: {title, form_id} }

    for m in MAPPINGS:
        old_id = m["old_id"]
        bucket = m["bucket"]
        pdf_path = ROOT / m["pdf"]
        expl_old = EXPLAINERS / bucket / f"{old_id}.json"

        if not expl_old.exists():
            errors.append(f"[MISS] Explainer not found: {expl_old}")
            continue
        if not pdf_path.exists():
            errors.append(f"[MISS] PDF not found: {pdf_path}")
            continue

        h = canonical_template_hash(pdf_path)

        expl = load_json(expl_old)

        # Prefer title from legacy registry if present; fallback to existing JSON title or old_id
        legacy_title = ""
        if isinstance(legacy_map.get(old_id), dict):
            legacy_title = str(legacy_map[old_id].get("title") or "")
        title = expl.get("title") or legacy_title or old_id

        # Build aliases set
        aliases = set()
        # from explainer file
        if isinstance(expl.get("aliases"), list):
            aliases.update([str(a) for a in expl["aliases"]])
        if "form_id" in expl:
            aliases.add(str(expl["form_id"]))
        # from legacy registry map (their "form_id" and the legacy key)
        if isinstance(legacy_map.get(old_id), dict):
            lf = legacy_map[old_id].get("form_id")
            if lf:
                aliases.add(str(lf))
        aliases.add(old_id)
        aliases.add(pdf_path.stem)

        # Stamp new fields
        expl["canonical_id"] = h    # template-based canonical id
        expl["bucket"] = expl.get("bucket", bucket)
        expl["schema_version"] = expl.get("schema_version", 1)
        expl.setdefault("created_at", now)
        expl["updated_at"] = now
        expl["aliases"] = sorted(list(aliases))

        # Save by hash
        new_path = EXPLAINERS / bucket / f"{h}.json"
        save_json(new_path, expl)

        # Update registry to the new schema
        upsert_registry(h, title, bucket, new_path, list(aliases))

        print(f"[OK] {old_id} → {h} → {new_path}")

    if errors:
        print("\nWARNINGS/ERRORS:")
        for e in errors:
            print(" -", e)

if __name__ == "__main__":
    main()
