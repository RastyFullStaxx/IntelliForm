import json, argparse, re
from collections import OrderedDict

CANON_MAP = [
    (r"\b(account|acct)\s*(no|number)\b", "ACCOUNT_NO"),
    (r"\baccount\s*type\b", "ACCOUNT_TYPE"),
    (r"\bbank\s*name\b", "BANK_NAME"),
    (r"\bbranch\s*name\b", "BRANCH_NAME"),
    (r"\bbranch\s*address\b", "BRANCH_ADDRESS"),
    (r"\b(check|cheque)\s*(no|number)\b", "CHECK_NO"),
    (r"\bdeposit\s*slip\s*(no|number)\b", "DEPOSIT_SLIP_NO"),
    (r"\bcard\s*(no|number)\b", "CARD_NO"),
    (r"\btin\b", "TIN"), (r"\bsss\b", "SSS_NO"), (r"\bgsis\b", "GSIS_NO"),
    (r"\bpag[-\s]*ibig\b", "PAGIBIG_NO"),
    (r"\bmonthly\s*income\b", "MONTHLY_INCOME"),
    (r"\binterest\s*rate\b", "INTEREST_RATE"),
    (r"\bterm\b", "TERM"), (r"\bloan\s*type\b", "LOAN_TYPE"),
    (r"\bemployer\s*name\b", "EMPLOYER_NAME"), (r"\bemployer\s*address\b", "EMPLOYER_ADDRESS"),
    (r"\bposition\b", "POSITION"), (r"\bcivil\s*status\b", "CIVIL_STATUS"),
    (r"\bnationality\b", "NATIONALITY"),
    (r"\bbirth\s*date\b|\bbirthdate\b|\bd\.?o\.?b\.?\b", "BIRTHDATE"),
    (r"\bmother('?s)?\s*maiden\s*name\b", "MOTHER_MAIDEN_NAME"),
    # DNFBP/AMLC extras
    (r"\bcif\s*(number|no\.?)\b", "CIF_NO"),
    (r"\bcustomer\s*name\b", "CUSTOMER_NAME"),
    (r"\bappropriate\s*government\s*agency\b|\baga\b", "AGA_NAME"),
    (r"\bdesignated\s*non[-\s]*financial\s*business\b|\bdnfbp\b", "DNFBP_TYPE"),
    (r"\bprimary\s*officer\b|\b20%\s*(ubo|ultimate\s*beneficial\s*owner)\b|\bposition/?s?\b", "RELATED_POSITION"),
    (r"\brelationship/?s?\b", "RELATED_RELATIONSHIP"),
    (r"\btype/?s?\s*of\s*transactions\b|\btransactions?\s*related\b", "TRANSACTION_DETAILS"),
    (r"\bamlc\b|\b(p|)cor\b|\bcertificate\s*of\s*registration\b|\bnot\s*required\b", "AMLC_REG_STATUS"),
    (r"\bjustification\b|\bnot\s*required\s*to\s*register\b", "AMLC_JUSTIFICATION"),
    (r"\bml/tf\s*prevention\s*program\b|\bmtpp\b|\bmanual\b", "MTPP_PRESENT"),
    (r"\bon[-\s]*boarding\b|\bkyc\b|\bcdd\b|\bongoing\s*monitoring\b|\bcovered\s*transaction\b|\bsuspicious\s*transaction\b|\brecord\s*keeping\b|\btraining\b|\baudits?\b", "MTPP_AREAS_TEXT"),
    (r"\bsignature\b", "SIGNATURE"),
    (r"\bdate\s*signed\b|\bmm/?dd/?yyyy\b", "DATE_SIGNED"),
]

def load_questions(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept dict-with-arrays OR list-of-dicts
    if isinstance(data, list):
        return [t.get("text","") for t in data if str(t.get("label","")).endswith("QUESTION")]
    tokens = data.get("tokens", [])
    labels = data.get("labels", ["O"]*len(tokens))
    return [tokens[i] for i,l in enumerate(labels) if str(l).endswith("QUESTION")]

def append_labels(labels_path, new_labels):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f, object_pairs_hook=OrderedDict)
    existing = set(labels.keys())
    added = []
    if new_labels:
        next_id = max(labels.values())
        for lb in new_labels:
            if lb not in existing:
                next_id += 1
                labels[lb] = next_id
                added.append(lb)
        if added:
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(labels, f, indent=2, ensure_ascii=False)
    return added

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--labels", default="data/labels_ph.json")
    args = ap.parse_args()

    qs = load_questions(args.json)
    found = set()
    lowqs = [q.lower() for q in qs]
    for t in lowqs:
        for rx, lb in CANON_MAP:
            if re.search(rx, t): found.add(lb)
    added = append_labels(args.labels, sorted(found))
    print("Proposals:", sorted(found))
    print("Added to labels_ph.json:", added)
