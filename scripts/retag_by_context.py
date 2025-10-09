import json, argparse, re
from math import hypot

QUESTION_TO_LABEL = [
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
    (r"\bposition\b|\b20%\s*(ubo|ultimate\s*beneficial\s*owner)\b|\bprimary\s*officer\b", "RELATED_POSITION"),
    (r"\brelationship\b", "RELATED_RELATIONSHIP"),
    (r"\btype/?s?\s*of\s*transactions\b|\btransactions?\s*related\b", "TRANSACTION_DETAILS"),
    (r"\bamlc\b|\b(p|)cor\b|\bcertificate\s*of\s*registration\b|\bnot\s*required\b|\bin\s*the\s*process\b", "AMLC_REG_STATUS"),
    (r"\bjustification\b|\bnot\s*required\s*to\s*register\b", "AMLC_JUSTIFICATION"),
    (r"\bml/tf\s*prevention\s*program\b|\bmtpp\b|\bmanual\b", "MTPP_PRESENT"),
    (r"\bon[-\s]*boarding\b|\bkyc\b|\bcdd\b|\bongoing\s*monitoring\b|\bcovered\s*transaction\b|\bsuspicious\s*transaction\b|\brecord\s*keeping\b|\btraining\b|\baudits?\b|\bregulatory\b", "MTPP_AREAS_TEXT"),
    (r"\bsignature\b|\bsignature\s*over\s*printed\s*name\b", "SIGNATURE"),
    (r"\bdate\s*signed\b|\bmm/?dd/?yyyy\b", "DATE_SIGNED"),
    (r"\bcif\s*(number|no\.?)\b", "CIF_NO"),
    (r"\bcustomer\s*name\b", "CUSTOMER_NAME"),
    (r"\bappropriate\s*government\s*agency\b|\baga\b", "AGA_NAME"),
    (r"\bdnfbp\b|\bdesignated\s*non[-\s]*financial\s*business\b", "DNFBP_TYPE"),
]

GENERIC_VALUE_LABELS = {"ID_NUMBER","AMOUNT","EMAIL","CONTACT_NO","ADDRESS","NAME","DATE","ANSWER"}

def load_triplets(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        # convert list-of-dicts → unified dict
        toks, boxes, labs = [], [], []
        for t in data:
            toks.append(t.get("text",""))
            boxes.append(t.get("bbox",[0,0,0,0]))
            labs.append(t.get("label","O"))
        return toks, boxes, labs, 1
    # dict-with-arrays
    toks = data.get("tokens", [])
    boxes = data.get("bboxes", [[0,0,0,0]]*len(toks))
    labs  = data.get("labels", ["O"]*len(toks))
    page  = data.get("page", 1)
    return toks, boxes, labs, page

def save_triplets(json_path, tokens, bboxes, labels):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"tokens":tokens,"bboxes":bboxes,"labels":labels}, f, indent=2, ensure_ascii=False)

def label_from_question(text):
    low = text.lower()
    for rx, target in QUESTION_TO_LABEL:
        if re.search(rx, low): return target
    return None

def group_spans(labels):
    spans = []
    i = 0
    n = len(labels)
    while i < n:
        lab = str(labels[i] or "O")
        if lab.startswith("B-"):
            base = lab[2:]
            j = i + 1
            while j < n and str(labels[j]) == f"I-{base}":
                j += 1
            spans.append((i, j, base))
            i = j
        elif lab.endswith("ANSWER"):  # tolerate naked I-ANSWER
            j = i + 1
            while j < n and str(labels[j]).endswith("ANSWER"):
                j += 1
            spans.append((i, j, "ANSWER"))
            i = j
        else:
            i += 1
    return spans

def center(box):
    x1,y1,x2,y2 = box
    return ( (x1+x2)/2.0, (y1+y2)/2.0 )

def nearest_question(tokens, bboxes, labels, idx, max_dist=120):
    cx, cy = center(bboxes[idx])
    best_k, best_d = None, 1e9
    for k,(t,l,b) in enumerate(zip(tokens,labels,bboxes)):
        if str(l).endswith("QUESTION"):
            dx,dy = center(b)
            d = hypot(dx-cx, dy-cy)
            if d < best_d:
                best_d, best_k = d, k
    return best_k if best_d <= max_dist else None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", help="optional output path; defaults to in-place")
    args = ap.parse_args()

    tokens, bboxes, labels, _ = load_triplets(args.json)

    # Build synthetic QUESTION text per token (concatenate nearby QUESTION tokens)
    # Simpler: use per-token text directly; regex will still match key words.
    upgrades = 0
    spans = group_spans(labels)

    for s,e,base in spans:
        if base in GENERIC_VALUE_LABELS or base in {"ID_NUMBER"}:
            nq = nearest_question(tokens, bboxes, labels, s)
            if nq is not None:
                target = label_from_question(tokens[nq])
                if target and target != base:
                    labels[s] = f"B-{target}"
                    for k in range(s+1, e):
                        labels[k] = f"I-{target}"
                    upgrades += 1

    outp = args.out or args.json
    save_triplets(outp, tokens, bboxes, labels)
    print(f"Upgraded spans: {upgrades}")
