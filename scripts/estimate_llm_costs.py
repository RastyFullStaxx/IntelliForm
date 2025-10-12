# scripts/estimate_llm_costs.py
from __future__ import annotations
import os, json, math, argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# local helpers (we'll reuse your snippet extractor & base dirs)
from scripts import config

# ---------- tokenization helpers ----------
def _load_tiktoken(model_hint: str = "gpt-4o-mini"):
    try:
        import tiktoken  # type: ignore
        try:
            enc = tiktoken.encoding_for_model(model_hint)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return enc
    except Exception:
        return None

def _count_tokens(text: str, enc=None) -> int:
    if not text:
        return 0
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # heuristic fallback: ~4 chars per token (safe-ish for planning)
    return max(1, math.ceil(len(text) / 4))

# ---------- estimation logic ----------
def estimate_pdf_tokens(
    pdf_path: Path,
    enc=None,
    prompt_overhead: int = 250,
    output_tokens_assumed: int = 600,
    snippet_chars: int = 4000,
) -> Tuple[int, int, int]:
    """
    Returns (input_tokens, output_tokens, total_tokens).
    - input_tokens: tokens from the PDF text snippet + prompt scaffolding
    - output_tokens: assumed average completion size per doc
    """
    snippet = config.quick_text_snippet(str(pdf_path), max_chars=snippet_chars)
    input_tokens = _count_tokens(snippet, enc=enc) + prompt_overhead
    output_tokens = output_tokens_assumed
    return input_tokens, output_tokens, input_tokens + output_tokens

def dollars_for_tokens(inp_toks: int, out_toks: int, price_in: float, price_out: float) -> float:
    return (inp_toks / 1000.0) * price_in + (out_toks / 1000.0) * price_out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Estimate LLM token & cost for PH supplemental forms")
    ap.add_argument("--dir", default=str(config.BASE_DIR / "uploads" / "ph-supplemental-forms"),
                    help="Folder with PDFs to estimate")
    ap.add_argument("--model", default=os.getenv("INTELLIFORM_ENGINE_MODEL", "gpt-4o-mini"),
                    help="Model hint for tokenizer mapping (affects tiktoken encoding choice)")
    ap.add_argument("--prompt-overhead", type=int, default=int(os.getenv("EST_PROMPT_OVERHEAD", "250")),
                    help="Estimated non-PDF prompt tokens (system rules, schema, etc.)")
    ap.add_argument("--assumed-output", type=int, default=int(os.getenv("EST_OUTPUT_TOKENS", "600")),
                    help="Assumed average output tokens per PDF")
    ap.add_argument("--snippet-chars", type=int, default=int(os.getenv("EST_SNIPPET_CHARS", "4000")),
                    help="Max chars to read from each PDF for estimation")
    ap.add_argument("--price-in", type=float, default=float(os.getenv("EST_PRICE_IN", "0.005")),
                    help="$/1k input tokens (set to your provider rate)")
    ap.add_argument("--price-out", type=float, default=float(os.getenv("EST_PRICE_OUT", "0.015")),
                    help="$/1k output tokens (set to your provider rate)")
    ap.add_argument("--save", action="store_true", help="Save NDJSON & summary JSON under explanations/logs")
    args = ap.parse_args()

    pdf_dir = Path(args.dir)
    if not pdf_dir.exists():
        print(f"[estimate] folder not found: {pdf_dir}")
        return

    enc = _load_tiktoken(args.model)

    pdfs = sorted([p for p in pdf_dir.glob("*.pdf")], key=lambda x: x.name)
    if not pdfs:
        print("[estimate] no PDFs found.")
        return

    print(f"[estimate] scanning {len(pdfs)} PDFs in {pdf_dir}")
    rows = []
    total_in = total_out = 0
    for i, pdf in enumerate(pdfs, 1):
        inp, outp, tot = estimate_pdf_tokens(
            pdf, enc=enc,
            prompt_overhead=args.prompt_overhead,
            output_tokens_assumed=args.assumed_output,
            snippet_chars=args.snippet_chars,
        )
        cost = dollars_for_tokens(inp, outp, args.price_in, args.price_out)
        rows.append({
            "pdf": str(pdf.relative_to(config.BASE_DIR)),
            "input_tokens": inp,
            "output_tokens": outp,
            "total_tokens": tot,
            "price_in_per_1k": args.price_in,
            "price_out_per_1k": args.price_out,
            "estimated_usd": round(cost, 4),
        })
        total_in += inp
        total_out += outp
        print(f"  [{i:02}/{len(pdfs):02}] {pdf.name}  in={inp:,}  out={outp:,}  ≈${cost:.4f}")

    grand = {
        "count": len(pdfs),
        "sum_input_tokens": total_in,
        "sum_output_tokens": total_out,
        "sum_total_tokens": total_in + total_out,
        "assumptions": {
            "prompt_overhead": args.prompt_overhead,
            "assumed_output_tokens": args.assumed_output,
            "snippet_chars": args.snippet_chars,
            "model_hint": args.model,
            "price_in_per_1k": args.price_in,
            "price_out_per_1k": args.price_out,
        },
        "estimated_total_usd": round(dollars_for_tokens(total_in, total_out, args.price_in, args.price_out), 4),
    }

    print("\n[estimate] SUMMARY")
    print(f"  Files: {grand['count']}")
    print(f"  Input tokens:  {grand['sum_input_tokens']:,}")
    print(f"  Output tokens: {grand['sum_output_tokens']:,}")
    print(f"  Total tokens:  {grand['sum_total_tokens']:,}")
    print(f"  Estimated cost: ${grand['estimated_total_usd']:.4f}")

    if args.save:
        logs_dir = config.EXPL_DIR / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ndjson_path = logs_dir / "ph-supplemental-cost-estimate.ndjson"
        with ndjson_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary_path = logs_dir / "ph-supplemental-cost-estimate.summary.json"
        summary_path.write_text(json.dumps(grand, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[estimate] wrote:\n  {ndjson_path}\n  {summary_path}")

if __name__ == "__main__":
    main()
