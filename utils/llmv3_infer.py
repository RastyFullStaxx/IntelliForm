"""
IntelliForm — Unified Inference Engine
======================================

WHAT THIS MODULE DOES
---------------------
Provides a high-level "one call" interface to run the complete IntelliForm
inference pipeline on PDFs or pre-extracted tokens:

1) **Extraction (optional)**: If given a PDF, calls `utils.extractor` to get
   tokens + bboxes (and page images if needed).
2) **Classification**: Loads `saved_models/classifier.pt` and runs the
   LayoutLMv3(+GNN)+Classifier to predict field labels per token.
3) **Grouping**: Aggregates token-level predictions into field-level groups
   (e.g., merge contiguous BIO segments, attach nearby values).
4) **Summarization**: Calls `utils.t5_summarize` to generate human-readable
   descriptions per detected field.
5) **Packaging**: Returns a structured result for UI or downstream evaluation.

PRIMARY ENTRYPOINTS
-------------------
- analyze_pdf(pdf_path: str, config: dict) -> dict
    End-to-end: PDF -> tokens/bboxes -> classify -> summarize -> result dict.

- analyze_tokens(tokens: List[str], bboxes: List[List[int]], config: dict) -> dict
    Skips extraction; starts from tokenized inputs.

OUTPUT FORMAT
-------------
{
  "document": "path_or_id",
  "fields": [
    {
      "label": "FULL_NAME",
      "score": 0.93,
      "tokens": ["John", "A.", "Doe"],
      "bbox": [x0,y0,x1,y1],           # merged span bbox
      "summary": "The user's full legal name.",
      "page": 1
    },
    ...
  ],
  "runtime": {"extract_ms": 120, "classify_ms": 85, "summarize_ms": 40}
}

CONFIG EXPECTATIONS
-------------------
- model paths: {"classifier": "saved_models/classifier.pt", "t5": "saved_models/t5.pt"}
- tokenizer/model names, device, batch sizes
- grouping thresholds (IOU merge, max gap), min_confidence, etc.

DEPENDENCIES
------------
- utils.field_classifier (loading + forward)
- utils.t5_summarize (summary generation)
- utils.extractor (optional, if starting from PDF)
- utils.graph_builder (if you compute edges at inference)
- utils.metrics (if computing on-the-fly evaluation against dev labels)

INTERACTIONS
------------
- Called by: `inference.py` (CLI), `api.py` (FastAPI endpoints), UI flows
- Returns: structured dict that the frontend can display or the evaluator can log

EXTENSION POINTS / TODOs
------------------------
- Add calibration of confidence (ECE) and show in UI.
- Add fallback summarization if T5 unavailable (template rules).
- Add per-page batching for very long documents.

"""


import torch
from transformers import LayoutLMv3Processor, LayoutLMv3Model
from PIL import Image
from utils.field_classifier import processor, model, predict_fields

# ⬇️ Optional model for generating raw embeddings (used outside classifier)
embedding_model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")

# 📦 Processor for encoding inputs without OCR
embedding_processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False
)

def normalize_bbox(bbox, width=1000, height=1000):
    """
    Normalize absolute bounding boxes to LayoutLMv3's 0–1000 scale.
    """
    x0, y0, x1, y1 = bbox
    return [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height)
    ]

def prepare_inputs(extracted_data, image):
    """
    Converts tokens and boxes into LayoutLMv3-compatible input for a single page.
    """
    words = [item['text'] for item in extracted_data]
    boxes = [normalize_bbox(item['bbox']) for item in extracted_data]

    encoding = embedding_processor(
        images=image,
        text=words,
        boxes=boxes,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )
    return encoding

def run_layoutlmv3_inference(model_inputs):
    """
    Runs LayoutLMv3 to get hidden state embeddings.
    """
    with torch.no_grad():
        outputs = embedding_model(**model_inputs)
    return outputs.last_hidden_state

def encode_with_layoutlmv3(tokens, bboxes, page_images):
    """
    Runs LayoutLMv3 across all pages to obtain raw embeddings.
    Used for GNNs or visualization tasks.
    """
    all_embeddings = []
    offset = 0

    for image in page_images:
        page_tokens, page_bboxes = [], []

        while offset < len(tokens) and len(page_tokens) < 512:
            page_tokens.append(tokens[offset])
            page_bboxes.append(bboxes[offset])
            offset += 1

        if not page_tokens:
            continue

        extracted_data = [{"text": t, "bbox": b} for t, b in zip(page_tokens, page_bboxes)]
        model_inputs = prepare_inputs(extracted_data, image)
        hidden_states = run_layoutlmv3_inference(model_inputs)

        seq_len = model_inputs["input_ids"].shape[1]
        all_embeddings.extend(hidden_states[0, :seq_len].detach().cpu().tolist())

    return all_embeddings

def classify_fields_from_pdf(filepath):
    """
    (Optional) Wrapper for direct LayoutLMv3 classification on raw PDF using OCR
    """
    encoding = processor(filepath, return_tensors="pt")
    results = predict_fields(encoding)
    return results
