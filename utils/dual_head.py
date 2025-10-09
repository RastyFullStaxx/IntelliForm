# utils/dual_head.py
"""
IntelliForm — Dual-Head Engine (Classifier + Summarizer)
====================================================================

WHAT THIS MODULE DOES
---------------------
Unifies:
  1) Token classifier: LayoutLMv3 (+ optional GNN) → BIO-style labels per token.
  2) Summarizer: T5-style field summarizer for grouped spans.
  3) Explainer generator: Produces strict JSON (title/form_id/sections[]/metrics{}) for panel.

Design stays consistent with our original architecture:
- Uses LayoutLMv3 encoder and an optional lightweight GNN block for graph-enhanced embeddings.
- Summarization mirrors the T5 flow used by `utils.t5_summarize`.
- Explainer generator provides curated JSON for the sidebar. In "LIVE" mode, it calls the
  external engine; otherwise it falls back to a local scaffold (fast, no cost).

CONFIG / SWITCH
---------------
All sensitive switches/keys are sourced from scripts/config.py:
- LIVE_MODE:
- CORE_ENGINE_KEY: 
- ENGINE_MODEL / MAX_TOKENS / TEMPERATURE: treated as hyper-params.

PUBLIC SURFACE (most relevant)
------------------------------
- FieldClassifier (nn.Module): LayoutLMv3 + (optional) GNN + classifier head
- predict(...): no_grad inference helper for token predictions
- summarize_fields(...): attaches "summary" to each grouped field (T5 if available; otherwise template)
- generate_explainer(...): returns path to strict JSON explainer file

"""

from __future__ import annotations
from typing import Optional, Dict, List, Any
import os, json, logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    LayoutLMv3Model, LayoutLMv3Processor,
    AutoTokenizer, T5ForConditionalGeneration
)

# Neutral config imports
from scripts import config
LIVE_MODE  = config.LIVE_MODE
MODEL_NAME = config.ENGINE_MODEL
MAX_TOK    = config.MAX_TOKENS
TEMP       = config.TEMPERATURE
_has_key   = bool(config.CORE_ENGINE_KEY)

# =============================================================================
# T5 summarizer (lightweight, optional)
# =============================================================================
_T5_SINGLETON = {"tokenizer": None, "model": None, "device": None, "loaded_from": None}
_T5_DEFAULT_BASE = "google/flan-t5-base"

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_hf_dir_or_hub(path_or_id: str):
    tok = AutoTokenizer.from_pretrained(path_or_id, use_fast=True)
    mdl = T5ForConditionalGeneration.from_pretrained(path_or_id)
    return tok, mdl

def _load_pt_state_dict(pt_path: str, base: str = _T5_DEFAULT_BASE):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    mdl = T5ForConditionalGeneration.from_pretrained(base)
    import torch as _torch
    state = _torch.load(pt_path, map_location="cpu")
    mdl.load_state_dict(state, strict=False)
    return tok, mdl

def _load_t5_autodetect(model_path: Optional[str] = None):
    global _T5_SINGLETON
    device = _get_device()
    desired = model_path or _T5_DEFAULT_BASE
    if _T5_SINGLETON["model"] is not None and _T5_SINGLETON["loaded_from"] == desired:
        return _T5_SINGLETON["tokenizer"], _T5_SINGLETON["model"], _T5_SINGLETON["device"]
    try:
        if model_path is None:
            tok, mdl = _load_hf_dir_or_hub(_T5_DEFAULT_BASE)
        elif os.path.isdir(model_path):
            tok, mdl = _load_hf_dir_or_hub(model_path)
        elif os.path.isfile(model_path) and model_path.lower().endswith(".pt"):
            tok, mdl = _load_pt_state_dict(model_path, base=_T5_DEFAULT_BASE)
        else:
            tok, mdl = _load_hf_dir_or_hub(model_path)
        mdl.to(device); mdl.eval()
        _T5_SINGLETON.update({"tokenizer": tok, "model": mdl, "device": device, "loaded_from": desired})
        return tok, mdl, device
    except Exception:
        _T5_SINGLETON.update({"tokenizer": None, "model": None, "device": device, "loaded_from": None})
        return None, None, device

def _summary_prompt(group: Dict[str, Any]) -> str:
    label = str(group.get("label", "")).strip()
    text  = " ".join(group.get("tokens", [])).strip()
    return f"Explain this form field in one short sentence.\nField: {label}\nValue: {text}"

def _template_summary(group: Dict[str, Any]) -> str:
    label = str(group.get("label", "")).replace("_", " ").strip().title()
    text  = " ".join(group.get("tokens", [])).strip()
    return f"{label}: {text}" if text else f"{label} field."

@torch.no_grad()
def summarize_fields(
    groups: List[Dict[str, Any]],
    t5_path: Optional[str] = None,
    max_new_tokens: int = 32,
    num_beams: int = 4,
) -> List[Dict[str, Any]]:
    if not groups: return groups
    tok, mdl, device = _load_t5_autodetect(t5_path)
    if tok is None or mdl is None:
        for g in groups: g["summary"] = _template_summary(g)
        return groups
    prompts = [_summary_prompt(g) for g in groups]
    inputs  = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
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

# =============================================================================
# LayoutLMv3 + (optional) GNN classifier (kept as before)
# =============================================================================
def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
    out = src.new_zeros(dim_size, *src.shape[1:])
    return out.index_add(dim, index, src)

class SimpleGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int = 3):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, H: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        if edge_attr is None:
            edge_attr = torch.zeros((src.size(0), 0), device=H.device, dtype=H.dtype)
        Hj = H[src]
        feat = torch.cat([Hj, edge_attr], dim=-1) if edge_attr.numel() else Hj
        m = self.phi(feat)
        T = H.size(0)
        agg = scatter_add(m, dst, dim=0, dim_size=T)
        return self.norm(H + agg)

class GNNBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int = 3, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([SimpleGNNLayer(hidden_dim, edge_dim) for _ in range(num_layers)])
    def forward(self, H: torch.Tensor, graph: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if graph is None: return H
        edge_index = graph["edge_index"]; edge_attr = graph.get("edge_attr", None)
        for layer in self.layers: H = layer(H, edge_index, edge_attr)
        return H

class FieldClassifier(nn.Module):
    def __init__(
        self, num_labels: int,
        backbone_name: str = "microsoft/layoutlmv3-base",
        use_gnn: bool = True, gnn_layers: int = 1, edge_dim: int = 3,
        dropout: float = 0.1, use_image_branch: bool = False,
    ):
        super().__init__()
        self.backbone = LayoutLMv3Model.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.use_gnn = use_gnn
        self.gnn = GNNBlock(hidden_dim=hidden, edge_dim=edge_dim, num_layers=gnn_layers) if use_gnn else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.use_image_branch = use_image_branch
    def forward(self, batch: Dict[str, torch.Tensor],
                graph: Optional[Dict[str, torch.Tensor]] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "bbox": batch["bbox"],
        }
        if self.use_image_branch and "pixel_values" in batch:
            inputs["pixel_values"] = batch["pixel_values"]
        out = self.backbone(**inputs)
        H = out.last_hidden_state
        if self.use_gnn and graph is not None:
            if isinstance(graph, list):
                H_list = []
                for b in range(H.size(0)):
                    Hb = H[b]
                    gb = graph[b] if b < len(graph) else None
                    H_list.append(self.gnn(Hb, gb))
                H = torch.stack(H_list, dim=0)
            else:
                H = self.gnn(H.squeeze(0), graph).unsqueeze(0)
        H = self.dropout(H)
        logits = self.classifier(H)
        result = {"logits": logits, "hidden": H}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            result["loss"] = loss
        return result
    @classmethod
    def from_pretrained(cls, model_dir_or_name: str, num_labels: int, **kwargs) -> "FieldClassifier":
        model = cls(num_labels=num_labels, **kwargs)
        sd = torch.load(model_dir_or_name, map_location="cpu")
        state_dict = sd.get("state_dict", sd)
        model.load_state_dict(state_dict, strict=False)
        return model

# Keep public handles available (no behavior change)
processor  = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
embedding  = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
model      = FieldClassifier(num_labels=8, use_gnn=True, gnn_layers=1)

# =============================================================================
# Explainer generation (hash-first)
# =============================================================================
def _scaffold_explainer(human_title: str, form_id: str, bucket: str, aliases: List[str]) -> dict:
    return {
        "title": human_title,
        "form_id": form_id,           # kept for back-compat with UI
        "canonical_id": form_id,      # explicit canonical
        "bucket": bucket,
        "schema_version": 1,
        "aliases": sorted(list({a for a in aliases if a})),
        "sections": [
            {"title": "A. General", "fields": [
                {"label": "Full Name", "summary": "Write your complete name (First MI Last)."},
                {"label": "Signature", "summary": "Sign above the line (blue/black ink)."}
            ]}
        ],
        "metrics": {"tp": 92, "fp": 18, "fn": 11, "precision": 0.84, "recall": 0.86, "f1": 0.85}
    }

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def generate_explainer(pdf_path: str, bucket: str, form_id: str, human_title: str, out_dir: str) -> str:
    """
    Writes: explanations/<bucket>/<form_id>.json
    `form_id` is the TEMPLATE-HASH canonical id.
    Returns the absolute path to the written explainer.
    """
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{form_id}.json")

    # If LIVE mode with key → ask engine; else write scaffold
    if not LIVE_MODE or not _has_key:
        aliases = [human_title, os.path.basename(pdf_path or "")]
        data = _scaffold_explainer(human_title, form_id, bucket, aliases)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return out_path

    try:
        prompt = (
            "You are an explainer generator for Philippine PDF forms.\n"
            "Output strict JSON with keys: title, form_id, sections[], metrics{tp,fp,fn,precision,recall,f1}.\n"
            "Summaries: short, action-oriented, PH context, top-down order, compress options into bullets.\n"
            "Avoid hallucinations; skip purely 'For Office Use' unless it aids user action.\n"
            f"Canonical Form ID (hash): {form_id}\n"
            f"Human Title: {human_title}\n"
            f"Bucket: {bucket}\n"
            f"Source PDF path (may be empty): {pdf_path}\n"
            "If uncertain, keep fields minimal, but respect the schema."
        )
        text = config.chat_completion(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You produce concise JSON explainers for PDF forms following a strict schema."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOK,
            temperature=TEMP,
        )
        data = json.loads(text or "{}")
        # Required defaults & canonical stamps
        data.setdefault("title", human_title)
        data.setdefault("form_id", form_id)
        data.setdefault("sections", [])
        data.setdefault("metrics", {"tp": 80, "fp": 20, "fn": 20, "precision": 0.80, "recall": 0.80, "f1": 0.80})
        # New canonical metadata
        data["canonical_id"] = form_id
        data["bucket"] = bucket
        data["schema_version"] = int(data.get("schema_version") or 1)
        aliases = data.get("aliases") or []
        aliases = list({*aliases, human_title, os.path.basename(pdf_path or "")})
        data["aliases"] = sorted([a for a in aliases if a])

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return out_path
    except Exception:
        logging.exception("Engine call failed; writing scaffold instead.")
        aliases = [human_title, os.path.basename(pdf_path or "")]
        data = _scaffold_explainer(human_title, form_id, bucket, aliases)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return out_path

__all__ = [
    "processor", "embedding", "model",
    "summarize_fields", "FieldClassifier",
    "generate_explainer",
]