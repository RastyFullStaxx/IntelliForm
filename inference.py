"""
IntelliForm — CLI Inference Runner
==================================

WHAT THIS MODULE DOES
---------------------
Provides a command-line entry point to run end-to-end IntelliForm inference
*without* the web UI. Useful for debugging and batch processing.

PIPELINE STEPS
--------------
1) If a PDF path is provided:
     - Call `utils.extractor` to obtain tokens, bboxes, (optionally) images.
2) Classification:
     - Load `saved_models/classifier.pt` via `utils.llmv3_infer` and predict token labels.
3) Grouping:
     - Merge BIO/contiguous tokens into field-level spans and compute span bboxes.
4) Summarization:
     - Use `utils.t5_summarize` (loaded by `llmv3_infer`) to produce natural-language summaries.
5) Output:
     - Print or save a JSON file with fields, labels, confidences, bboxes, and summaries.
     - Optionally write `static/metrics_report.txt` if metrics are computed.

TYPICAL USAGE
-------------
$ python inference.py --pdf "uploads/sample.pdf" \
    --classifier "saved_models/classifier.pt" \
    --t5 "saved_models/t5.pt" \
    --out "outputs/sample_result.json"

ARGUMENTS (SUGGESTED)
---------------------
--pdf <path>            Path to input PDF (or use --tokens_json for pre-extracted)
--tokens_json <path>    Skip extraction; provide tokens+bboxes JSON
--classifier <path>     Path to trained classifier weights
--t5 <path>             Path to trained T5 weights
--device <cpu|cuda>     Device selection
--min_conf <float>      Confidence threshold for keeping predictions
--save_report           If set, write `static/metrics_report.txt` (optional)
--out <path>            Where to save the structured JSON result

OUTPUT FORMAT
-------------
{
  "document": "uploads/sample.pdf",
  "fields": [
    {"label":"EMAIL","score":0.91,"tokens":["user","@","mail",".","com"],"bbox":[...],"summary":"...","page":1},
    ...
  ],
  "runtime": {"extract_ms":..., "classify_ms":..., "summarize_ms":...}
}

INTERACTIONS
------------
- Calls: utils.llmv3_infer.analyze_pdf / analyze_tokens
- Consumes: saved_models/classifier.pt, saved_models/t5.pt
- Produces: outputs/*.json (if --out is used)

NOTES & TODOs
-------------
- For bulk runs, support `--input_dir` and iterate over PDFs.
- Add CSV export if needed by downstream tools.

"""


from utils.extractor import extract_layout_data
from utils.llmv3_infer import prepare_inputs
from utils.field_classifier import predict_fields
from utils.t5_summarize import summarize_label
from utils.metrics import evaluate_field_predictions, evaluate_summaries


def run_inference(pdf_path: str, ground_truth_path: str = None, output_metrics_path: str = "static/metrics_report.txt"):
    """
    Orchestrates the full IntelliForm processing pipeline on the given PDF.
    Optionally computes evaluation metrics if ground truth is provided.

    Args:
        pdf_path (str): Path to the uploaded PDF file.
        ground_truth_path (str, optional): Path to ground truth JSON for evaluation.
        output_metrics_path (str, optional): Path to save metrics report.

    Returns:
        list[dict]: Annotated fields with labels, summaries, bboxes, confidence, and page number.
    """

    # Step 1: Extract layout-aware text, positions, and page-level images
    tokens, bboxes, page_nums, page_images = extract_layout_data(pdf_path)

    print(f"🧾 Tokens extracted: {len(tokens)}")
    print(f"🖼️ Pages found: {len(page_images)}")

    # Step 2: Group tokens and bboxes by page number
    pagewise_data = {}
    for text, bbox, page_num in zip(tokens, bboxes, page_nums):
        if page_num not in pagewise_data:
            pagewise_data[page_num] = []
        pagewise_data[page_num].append({"text": text, "bbox": bbox})

    # Step 3: Inference per page
    results = []

    for i, page_image in enumerate(page_images):
        page_num = i + 1
        if page_num not in pagewise_data:
            print(f"⚠️ No tokens found for page {page_num}, skipping.")
            continue

        print(f"🔄 Processing Page {page_num}...")

        model_inputs = prepare_inputs(pagewise_data[page_num], page_image)
        predicted_fields = predict_fields(model_inputs)

        for field in predicted_fields:
            label = field["label"]
            try:
                summary = summarize_label(label)
            except Exception as e:
                print(f"⚠️ Summary error for label '{label}': {e}")
                summary = "Summary unavailable"

            results.append({
                "label": label,
                "summary": summary,
                "bbox": field["bbox"],
                "confidence": field["confidence"],
                "text": field.get("text", ""),
                "page_num": page_num
            })

    # Step 4: Compute metrics if ground truth is available
    if ground_truth_path:
        print("📊 Computing evaluation metrics...")
        field_metrics = compute_field_metrics(results, ground_truth_path)
        summary_metrics = compute_summary_metrics(results, ground_truth_path)
        save_metrics_to_txt(output_metrics_path, field_metrics, summary_metrics)

    return results
