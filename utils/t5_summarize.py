# utils/t5_summarize.py

"""
IntelliForm — T5 Summarizer
===========================

WHAT THIS MODULE DOES
---------------------
Serves a T5 sequence-to-sequence model that generates concise, human-readable
summaries for grouped fields (after BIO grouping). Provides:
- Lazy model loading (once per process)
- Batched generation for speed
- Graceful fallback to rule-based summaries if T5 is unavailable
- **Auto-detects** a Hugging Face directory vs a raw `.pt` state_dict

WHEN IT'S USED
--------------
- Inference: called by `utils.llmv3_infer._summarize()` to attach summaries.
- Training: `scripts/train_t5.py` fine-tunes and saves weights under `saved_models/t5` (dir).

PRIMARY INPUTS
--------------
- groups: List[dict] where each dict contains:
    {
      "label": str,           # e.g., "FULL_NAME" or "NAME"
      "tokens": List[str],    # merged tokens for the span
      "bbox": [x0,y0,x1,y1],
      "page": int,
      "score": float
    }
- t5_path: Optional[str] to a fine-tuned checkpoint
  (can be HF ID, local dir, or a raw .pt state_dict file).

OUTPUTS
-------
- The same list of groups, each with an added "summary": str

KEY FUNCTIONS / CLASSES
-----------------------
- summarize_fields(groups, t5_path=None, **gen_kwargs) -> List[dict]
- T5Summarizer: generate(), batch_generate(), save_pretrained(), from_pretrained()

DEPENDENCIES
------------
- transformers: T5ForConditionalGeneration, AutoTokenizer
- torch

NOTES
-----
- Default base is "google/flan-t5-base".
- If `t5_path` is a `.pt` file, we load the default base then apply the `state_dict`.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# ------------------------------
# Globals for lazy loading
# ------------------------------
_T5_SINGLETON = {
    "tokenizer": None,
    "model": None,
    "device": None,
    "loaded_from": None,
}

_DEFAULT_BASE = "google/flan-t5-base"


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_hf_dir_or_hub(path_or_id: str) -> Tuple[Optional[AutoTokenizer], Optional[T5ForConditionalGeneration]]:
    tok = AutoTokenizer.from_pretrained(path_or_id, use_fast=True)
    mdl = T5ForConditionalGeneration.from_pretrained(path_or_id)
    return tok, mdl


def _load_pt_state_dict(pt_path: str, base: str = _DEFAULT_BASE) -> Tuple[Optional[AutoTokenizer], Optional[T5ForConditionalGeneration]]:
    # Load a base model/tokenizer, then load the raw state_dict
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    mdl = T5ForConditionalGeneration.from_pretrained(base)
    state = torch.load(pt_path, map_location="cpu")
    # allow missing keys if head sizes differ; user fine-tuned same base ideally
    mdl.load_state_dict(state, strict=False)
    return tok, mdl


def _load_t5_autodetect(model_path: Optional[str] = None):
    """
    Lazy-load T5/tokenizer with auto-detect:
      - None -> default HF base
      - existing directory -> HF directory load
      - existing .pt file -> load base then state_dict
      - otherwise -> treat as HF Hub ID
    """
    global _T5_SINGLETON
    device = _get_device()

    # Decide desired source string for caching key
    if model_path:
        desired = model_path
    else:
        desired = _DEFAULT_BASE

    # Cache hit
    if _T5_SINGLETON["model"] is not None and _T5_SINGLETON["loaded_from"] == desired:
        return _T5_SINGLETON["tokenizer"], _T5_SINGLETON["model"], _T5_SINGLETON["device"]

    # Try loads in order
    try:
        if model_path is None:
            tok, mdl = _load_hf_dir_or_hub(_DEFAULT_BASE)
        elif os.path.isdir(model_path):
            tok, mdl = _load_hf_dir_or_hub(model_path)
        elif os.path.isfile(model_path) and model_path.lower().endswith(".pt"):
            tok, mdl = _load_pt_state_dict(model_path, base=_DEFAULT_BASE)
        else:
            # hub id or local dir that doesn't exist yet -> let HF handle it
            tok, mdl = _load_hf_dir_or_hub(model_path)

        mdl.to(device)
        mdl.eval()
        _T5_SINGLETON.update({"tokenizer": tok, "model": mdl, "device": device, "loaded_from": desired})
        return tok, mdl, device

    except Exception:
        # mark unavailable; caller will template-fallback
        _T5_SINGLETON.update({"tokenizer": None, "model": None, "device": device, "loaded_from": None})
        return None, None, device


# ------------------------------
# Prompting utilities
# ------------------------------
def _make_prompt(group: Dict[str, Any]) -> str:
    """
    Build a stable, compact prompt from a field group.
    """
    label = str(group.get("label", "")).strip()
    text = " ".join(group.get("tokens", [])).strip()
    prompt = f"Explain this form field in one short sentence.\nField: {label}\nValue: {text}"
    return prompt


def _template_summary(group: Dict[str, Any]) -> str:
    """Fallback summary when T5 is unavailable."""
    label = str(group.get("label", "")).replace("_", " ").strip().title()
    text = " ".join(group.get("tokens", [])).strip()
    if text:
        return f"{label}: {text}"
    return f"{label} field."


# ------------------------------
# Public API
# ------------------------------
class T5Summarizer:
    """
    Thin wrapper around T5 for single/batch generation, with save/load helpers.
    """
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = "auto",
        max_new_tokens: int = 32,
        num_beams: int = 4,
    ):
        tok, mdl, dev = _load_t5_autodetect(model_name_or_path)
        if tok is None or mdl is None:
            raise RuntimeError("T5 model could not be loaded.")
        self.tokenizer = tok
        self.model = mdl
        self.device = _get_device() if device == "auto" else torch.device(device or "cpu")
        # ensure on desired device
        if self.device.type != dev.type:
            self.model.to(self.device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    @classmethod
    def from_pretrained(cls, model_dir_or_id: str, **kwargs) -> "T5Summarizer":
        return cls(model_name_or_path=model_dir_or_id, **kwargs)

    def save_pretrained(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self.model.save_pretrained(dir_path)
        self.tokenizer.save_pretrained(dir_path)

    @torch.no_grad()
    def generate(self, text: str, **overrides) -> str:
        max_new_tokens = int(overrides.get("max_new_tokens", self.max_new_tokens))
        num_beams = int(overrides.get("num_beams", self.num_beams))
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def batch_generate(self, texts: List[str], **overrides) -> List[str]:
        if not texts:
            return []
        max_new_tokens = int(overrides.get("max_new_tokens", self.max_new_tokens))
        num_beams = int(overrides.get("num_beams", self.num_beams))
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
        return [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]


def summarize_fields(
    groups: List[Dict[str, Any]],
    t5_path: Optional[str] = None,
    max_new_tokens: int = 32,
    num_beams: int = 4,
) -> List[Dict[str, Any]]:
    """
    Attach a "summary" string to each group. Uses T5 if available; otherwise falls back.

    Args:
      groups         : list of field groups (mutated with 'summary' key)
      t5_path        : local dir, HF model id, or .pt state_dict path
      max_new_tokens : generation length
      num_beams      : beam search width

    Returns:
      groups with "summary" added.
    """
    if not groups:
        return groups

    tok, mdl, device = _load_t5_autodetect(t5_path)
    if tok is None or mdl is None:
        for g in groups:
            g["summary"] = _template_summary(g)
        return groups

    prompts = [_make_prompt(g) for g in groups]
    with torch.no_grad():
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            num_beams=int(num_beams),
            early_stopping=True,
        )
        decoded = [tok.decode(o, skip_special_tokens=True).strip() for o in outputs]

    for g, s in zip(groups, decoded):
        g["summary"] = s if s else _template_summary(g)

    return groups
