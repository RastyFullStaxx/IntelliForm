# utils/t5_summarize.py

"""
IntelliForm — T5 Summarizer
===========================

WHAT THIS MODULE DOES
---------------------
Fine-tunes and serves a T5 sequence-to-sequence model that generates
human-readable summaries for disambiguated field labels (or grouped tokens).
Example:
  Input  : "Field: Date of Visit; Type: date"
  Output : "The date when the visit occurred."

WHEN IT'S USED
--------------
- **Training**: called by `scripts/train_t5.py` with (input_text, target_summary) pairs
  from `data/*/summaries.csv`.
- **Inference**: called by `utils/llmv3_infer.py` after classification groups tokens
  into fields; produces summaries for each field.

PRIMARY INPUTS
--------------
Training:
  - input_text: str (templated feature string built upstream)
  - target_summary: str (ground-truth natural-language description)
Inference:
  - input_text: str (from predicted field segments, e.g., label name + nearby context)

OUTPUTS
-------
- generate_summary(text: str, **gen_kwargs) -> str
- batch_generate(texts: List[str], **gen_kwargs) -> List[str]
- load/save utilities for model + tokenizer

KEY FUNCTIONS / CLASSES
-----------------------
- class T5Summarizer:
    __init__(model_name_or_path: str = "t5-small", device: str = "auto", **kwargs)
        Loads T5 + tokenizer; configures generation defaults (max_len, num_beams).

    fit(dataloader, optimizer, scheduler, log_fn=None) -> dict
        (Training loop is usually handled in `scripts/train_t5.py`, but an optional
         helper can live here if you want a self-contained trainer.)

    generate(text: str, **gen_kwargs) -> str
        Single-text generation.

    batch_generate(texts: List[str], **gen_kwargs) -> List[str]
        Batched generation.

    save_pretrained(dir_path)
    from_pretrained(dir_path) -> "T5Summarizer"

DEPENDENCIES
------------
- transformers: T5ForConditionalGeneration, T5Tokenizer
- torch, torch.utils.data
- pandas (only in training scripts that read CSVs)

INTERACTIONS
------------
- Called by: scripts/train_t5.py (training), utils/llmv3_infer.py (inference)
- Inputs created by: post-classification grouping logic (inference.py or llmv3_infer.py)

TRAINING NOTES
--------------
- Use Teacher Forcing via target labels.
- Loss is cross-entropy over decoder outputs.
- Save best checkpoint by validation ROUGE-L/METEOR.

INFERENCE NOTES
---------------
- Keep input template stable (e.g., "Field: {name}; Type: {type}; Context: {snippet}")
  so the model learns/uses consistent patterns.

EXTENSION POINTS / TODOs
------------------------
- Add prompt templates per domain (medical/shipping/etc.).
- Add constrained decoding for terms of art if needed.

"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 🔄 Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize_label(label_text: str, max_length: int = 20) -> str:
    """
    Generates a natural-language summary from a structured label.

    Args:
        label_text (str): A short field label (e.g., 'B-Email', 'I-Name', 'TIN')
        max_length (int): Max tokens to include in the output summary

    Returns:
        str: Summary sentence describing the field
    """
    prompt = f"Expand this form field label: {label_text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary
