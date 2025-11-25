"""
FIELD-LABEL DISAMBIGUATION AND COMPLETION-TIME REDUCTION IN PDF FORMS
USING LAYOUTLMV3 WITH GRAPH NEURAL NETWORK ARCHITECTURE
======================================================================

A Thesis Presented to the Faculty of the
College of Computer and Information Sciences, Polytechnic University of the Philippines
In Partial Fulfilment of the Requirements for the Degree
Bachelor of Science in Computer Science

Researchers:
  - Dinglasa, Roanne Maye B.
  - Espartero, Rasty C.
  - Mahayag, David Geisler M.
  - Placente, Yesa V.

Study Focus (Problem Statement):
  - Resolve semantic ambiguity across heterogeneous form field labels in noisy, multi-page PDFs.
  - Accurately localize field labels and values with LayoutLMv3 + GNN spatial reasoning.
  - Generate concise, context-aware field descriptions to assist end users.
  - Measure impact on user efficiency via completion-time reduction and precision/recall/F1, 
        plus ROUGE-L/METEOR for description quality.

Core Components
--------
1) Extraction (pdfplumber) → tokens + normalized layout boxes
2) Edge construction (graph builder) for the GNN layer
3) Encoder + GNN + classifier head (LayoutLMv3 backbone)
4) Summarization (T5-based decoder) for human-friendly field blurbs
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pdfplumber
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import LayoutLMv3Model, LayoutLMv3Processor, AutoTokenizer, T5ForConditionalGeneration


# -----------------------------------------------------------------------------
# 1) Extraction (pdfplumber + layout normalization)
# -----------------------------------------------------------------------------
"""
Program Title: 
  - (1) Extraction (pdfplumber + layout normalization)
  
Programmers: 
  - Dinglasa, Roanne Maye B.
  - Espartero, Rasty C.
  - Mahayag, David Geisler M.
  - Placente, Yesa V.
  
Where the program fits in the general system designs: 
  - Ingests PDFs and normalizes layout features for downstream encoder/GNN.
  
Date written: August 2025
Date revised: October 2025

Purpose: 
  - Convert raw PDF pages into a clean, layout-aware representation that can be
    consumed directly by downstream models (LayoutLMv3 encoder and GNN layer).
  - Standardize all geometric information into a shared 0–1000 coordinate space
    so every subsequent module can reason about spatial relationships consistently.
  - Encapsulate extracted content in simple dataclasses (pages, tokens) to make
    debugging, unit testing, and further preprocessing straightforward.
    
Data structures, algorithms, and control: 
  - Data structures:
      - `PageInfo` dataclass: stores page-level metadata (width, height, page_num)
        used to normalize coordinates and reason about per-page layouts.
      - `Token` dataclass: holds each word’s text, normalized bounding box, and
        page index; this is the fundamental unit passed into the encoder/GNN.
      - `ExtractResult` dataclass: aggregates all pages and tokens, and provides
        both a flat token list and a per-page grouping for later modules.
  - Algorithms:
      - Use `pdfplumber` to extract word-level text and raw PDF coordinates
        (x0, y0, x1, y1) from each page.
      - Apply `_normalize_bbox` to map raw coordinates into a 0–1000 LayoutLM-style
        grid, handling clamping and ordering to avoid invalid boxes.
      - Optionally de-duplicate whitespace and discard very short tokens to reduce
        noise before classification and graph construction.
  - Control:
      - Iterate pages sequentially, extracting words and building tokens_page by
        page to avoid excessive memory usage on long documents.
      - Immediately raise `FileNotFoundError` when the input PDF path is invalid
        so calling code can handle the error early.
      - Return a single `ExtractResult` object, which becomes the canonical input
        for the rest of the IntelliForm pipeline (encoder, GNN, summarizer, metrics).
"""
@dataclass
class Token:
    text: str
    bbox: Tuple[int, int, int, int]
    page: int


@dataclass
class PageInfo:
    width: float
    height: float
    page_num: int


@dataclass
class ExtractResult:
    pages: List[PageInfo]
    tokens: List[Token]
    tokens_by_page: List[List[Token]]


def _normalize_bbox(x0: float, y0: float, x1: float, y1: float,
                    width: float, height: float) -> Tuple[int, int, int, int]:
    """
    Normalize absolute PDF coordinates into 0–1000 LayoutLM-style range so encoder
    and graph layers share a consistent spatial scale.
    """
    def clamp(v: float, lo: int = 0, hi: int = 1000) -> int:
        return max(lo, min(hi, int(round(v))))

    nx0 = (x0 / width) * 1000
    ny0 = (y0 / height) * 1000
    nx1 = (x1 / width) * 1000
    ny1 = (y1 / height) * 1000
    nx0, nx1 = min(nx0, nx1), max(nx0, nx1)
    ny0, ny1 = min(ny0, ny1), max(ny0, ny1)
    return clamp(nx0), clamp(ny0), clamp(nx1), clamp(ny1)


def extract_pdf(pdf_path: str,
                dedupe_whitespace: bool = True,
                min_len: int = 1) -> ExtractResult:
    """
    Parse a PDF with pdfplumber and emit layout-normalized tokens/pages used by
    LayoutLMv3 + GNN. This is the ingestion step shown at the top of the pipeline.

    Args:
        pdf_path: path to the PDF file (ingestion source from UI upload)
        dedupe_whitespace: collapse repeated spaces inside words for cleaner BIO tags
        min_len: minimum token length to retain to avoid noise tokens
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: List[PageInfo] = []
    tokens_flat: List[Token] = []
    tokens_per_page: List[List[Token]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            width, height = float(page.width), float(page.height)
            pages.append(PageInfo(width=width, height=height, page_num=idx))

            words = page.extract_words(x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False)
            page_tokens: List[Token] = []

            for w in words or []:
                text = w.get("text", "")
                if dedupe_whitespace:
                    text = " ".join(text.split())
                if len(text) < min_len:
                    continue

                x0, x1 = float(w["x0"]), float(w["x1"])
                top, bottom = float(w["top"]), float(w["bottom"])
                bbox = _normalize_bbox(x0, top, x1, bottom, width, height)

                tok = Token(text=text, bbox=bbox, page=idx)
                page_tokens.append(tok)
                tokens_flat.append(tok)

            tokens_per_page.append(page_tokens)

    return ExtractResult(pages=pages, tokens=tokens_flat, tokens_by_page=tokens_per_page)


# -----------------------------------------------------------------------------
# 2) Graph construction (node/edge builder for GNN)
# -----------------------------------------------------------------------------
"""
Program Title:
  - (2) Graph construction (node/edge builder for GNN)
  
Programmers:
  - Dinglasa, Roanne Maye B.
  - Espartero, Rasty C.
  - Mahayag, David Geisler M.
  - Placente, Yesa V.
  
Where the program fits in the general system designs:
  - Creates spatial neighbor edges feeding the GNN layout layer.
  
Date written: August 2025
Date revised: October 2025

Purpose:
  - Provide a consistent graph representation of spatial relationships between
    tokens so the GNN can propagate information along “nearby” nodes.
  - Support both k-NN and radius-based neighborhood construction while keeping
    edges restricted within the same page when requested.
  - Output graph tensors (`edge_index`, `edge_attr`, `num_nodes`) in a format
    directly consumable by the GNN block in the classifier.

Data structures, algorithms, and control:
  - Data structures:
      - Input `bboxes: np.ndarray[T, 4]` containing normalized token boxes
        (x0, y0, x1, y1) in the 0–1000 LayoutLM coordinate space.
      - Optional `page_ids: np.ndarray[T]` marking which page each token belongs
        to, used to prevent edges from crossing page boundaries.
      - Output dict with:
          - `edge_index: torch.LongTensor[2, E]` listing directed edges
            (source → target node indices).
          - `edge_attr: torch.FloatTensor[E, 3]` storing per-edge features
            `[distance, dx, dy]` for geometry-aware message passing.
          - `num_nodes: int` representing the number of tokens/nodes T.
  - Algorithms:
      - Compute token centers `(cx, cy)` from each bounding box to serve as the
        basis for distance and neighbor calculations.
      - For `"knn"` strategy:
          - Use `scipy.spatial.KDTree` when available for efficient nearest
            neighbor queries; otherwise fall back to a brute-force distance matrix.
          - For each node, query its k nearest neighbors (excluding itself) and
            build edges `(i → j)` with associated distance and delta offsets.
      - For `"radius"` strategy:
          - Compute pairwise distances and connect all nodes within a specified
            radius, again excluding self-edges.
      - Apply `page_ids` filtering (when provided) so only tokens on the same
        page are connected, matching the visual layout assumption of the model.
      - If no edges are produced (e.g., very small T), fall back to a simple
        sequential chain `0→1→2→…` to keep the GNN graph well-defined.
  - Control:
      - Validate the `strategy` argument and raise a `ValueError` for unknown
        strategies to fail fast during configuration errors.
      - Keep graph construction deterministic for a given set of bboxes and
        parameters (no random sampling of neighbors).
      - Return the packed edge tensors in a single dictionary that upstream
        code can pass directly into the GNN-enhanced classifier.
"""

def _centers(bboxes: np.ndarray) -> np.ndarray:
    """Compute centers (x, y) from [x0, y0, x1, y1] boxes."""
    return np.stack([(bboxes[:, 0] + bboxes[:, 2]) / 2.0,
                     (bboxes[:, 1] + bboxes[:, 3]) / 2.0], axis=1)


def build_edges(bboxes: np.ndarray,
                strategy: str = "knn",
                k: int = 8,
                radius: Optional[float] = None,
                page_ids: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
    """
    Construct token adjacency edges for GNN message passing, mirroring the
    "edge construction (spatial neighbours)" box in the architecture:
      - k-NN (default) for stable fan-out
      - radius for dense local context
    Page IDs can be provided to prevent cross-page links.
    """
    from scipy.spatial import KDTree  # optional; falls back to brute force if missing

    num_nodes = bboxes.shape[0]
    centers = _centers(bboxes)
    edge_list: List[Tuple[int, int, float, float, float]] = []

    if strategy == "knn":
        try:
            tree = KDTree(centers)
            for i, c in enumerate(centers):
                dists, idxs = tree.query(c, k=k + 1)  # +1 = self
                for j, d in zip(idxs[1:], dists[1:]):  # skip self
                    if page_ids is not None and page_ids[i] != page_ids[j]:
                        continue
                    edge_list.append((i, j, float(d), float(centers[j][0] - c[0]), float(centers[j][1] - c[1])))
        except Exception:
            dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            for i in range(num_nodes):
                idxs = np.argsort(dmat[i])[:k + 1]
                for j in idxs:
                    if i == j:
                        continue
                    if page_ids is not None and page_ids[i] != page_ids[j]:
                        continue
                    d = float(dmat[i, j])
                    edge_list.append((i, j, d, float(centers[j][0] - centers[i][0]), float(centers[j][1] - centers[i][1])))

    elif strategy == "radius":
        assert radius is not None, "radius must be set when using radius strategy"
        dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        for i in range(num_nodes):
            neighbors = np.where(dmat[i] <= radius)[0]
            for j in neighbors:
                if i == j:
                    continue
                if page_ids is not None and page_ids[i] != page_ids[j]:
                    continue
                d = float(dmat[i, j])
                edge_list.append((i, j, d, float(centers[j][0] - centers[i][0]), float(centers[j][1] - centers[i][1])))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if not edge_list:
        edge_list = [(i, i + 1, 1.0, 1.0, 0.0) for i in range(num_nodes - 1)]

    edge_index = torch.tensor([[i, j] for (i, j, *_)
                               in edge_list], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([[d, dx, dy] for (_, _, d, dx, dy)
                              in edge_list], dtype=torch.float)
    return {"edge_index": edge_index, "edge_attr": edge_attr, "num_nodes": num_nodes}


# -----------------------------------------------------------------------------
# 3) Encoder + GNN + classifier (LayoutLMv3 backbone)
# -----------------------------------------------------------------------------
"""
Program Title:
  - IntelliForm Pipeline (Encoder + GNN Classifier)

Programmers:
  - Dinglasa, Roanne Maye B.
  - Espartero, Rasty C.
  - Mahayag, David Geisler M.
  - Placente, Yesa V.

Where the program fits in the general system designs:
  - Core transformer fusion + optional GNN contextualization + classifier head.

Date written: August 2025
Date revised: October 2025

Purpose:
  - Produce token-level field label logits that combine LayoutLMv3 contextual
    embeddings with graph-based spatial reasoning.
  - Offer a single `FieldClassifier` module that both training and inference
    code can call to obtain logits (and loss when labels are provided).

Data structures, algorithms, and control:
  - Data structures:
      - Batch dict: `input_ids`, `attention_mask`, `bbox`, optional `pixel_values`.
      - Optional graph dict: `edge_index`, `edge_attr`, `num_nodes` per example.
      - Model parts: LayoutLMv3 backbone, GNNBlock, linear classifier head.
  - Algorithms:
      - Run LayoutLMv3 to get `last_hidden_state` for each token.
      - If a graph is provided, apply one or more SimpleGNN layers to inject
        neighbor information into token embeddings.
      - Apply dropout, then a linear layer to map embeddings to label logits;
        optionally compute cross-entropy loss against token labels.
  - Control:
      - Support both single-graph (B=1) and list-of-graphs (batched) usage.
      - Keep the image branch optional via `use_image_branch`.
      - Provide `from_pretrained(...)` for restoring saved classifier weights.
"""

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
    """Index-based aggregation without external PyG dependency (keeps GNN lightweight)."""
    out = src.new_zeros(dim_size, *src.shape[1:])
    return out.index_add(dim, index, src)


class SimpleGNNLayer(nn.Module):
    """
    Edge-conditioned message passing (Graphical Neural Network Layer):
      m_ij = phi([h_j, e_ij])           # fuse neighbor hidden state with edge geom
      h_i' = LayerNorm(h_i + Σ_j m_ij)  # residual + spatially aware aggregation
    """
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

        T, _ = H.shape
        agg = scatter_add(m, dst, dim=0, dim_size=T)
        return self.norm(H + agg)


class GNNBlock(nn.Module):
    """Stack of SimpleGNNLayer(s) to enrich LayoutLMv3 embeddings with spatial context."""
    def __init__(self, hidden_dim: int, edge_dim: int = 3, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([SimpleGNNLayer(hidden_dim, edge_dim) for _ in range(num_layers)])

    def forward(self, H: torch.Tensor, graph: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if graph is None:
            return H
        edge_index = graph["edge_index"]
        edge_attr = graph.get("edge_attr", None)
        for layer in self.layers:
            H = layer(H, edge_index, edge_attr)
        return H


class FieldClassifier(nn.Module):
    """
    LayoutLMv3 encoder → optional GNN → linear classifier head.
    Represents the "Dual Head Architecture (Classifier)" in the diagram.
    """
    def __init__(self,
                 num_labels: int,
                 backbone_name: str = "microsoft/layoutlmv3-base",
                 use_gnn: bool = True,
                 gnn_layers: int = 1,
                 edge_dim: int = 3,
                 dropout: float = 0.1,
                 use_image_branch: bool = False):
        super().__init__()
        self.backbone = LayoutLMv3Model.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size

        self.use_gnn = use_gnn
        self.gnn = GNNBlock(hidden_dim=hidden, edge_dim=edge_dim, num_layers=gnn_layers) if use_gnn else None

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.use_image_branch = use_image_branch

    def forward(self,
                batch: Dict[str, torch.Tensor],
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
        H = out.last_hidden_state  # [B, T, D]

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

        result: Dict[str, torch.Tensor] = {"logits": logits, "hidden": H}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            result["loss"] = loss
        return result


# Primary processor and embedding used across the pipeline
embedding_processor: LayoutLMv3Processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False,
)
embedding = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
classifier_model = FieldClassifier(num_labels=8, use_gnn=True, gnn_layers=1)


# -----------------------------------------------------------------------------
# 4) End-to-end inference helpers (tokens / PDF)
# -----------------------------------------------------------------------------
"""
Program Title:
  - IntelliForm Pipeline (Inference Helpers)
Programmers:
  - Dinglasa, Roanne Maye B.
  - Espartero, Rasty C.
  - Mahayag, David Geisler M.
  - Placente, Yesa V.
Where the program fits in the general system designs:
  - Connects extraction/encoding/classification/summarization into single-call routines.
Date written and revised: 2024-09
Purpose:
  - Run analyze_tokens/analyze_pdf and return structured JSON with runtimes.
Data structures, algorithms, and control:
  - token arrays + bboxes → encoder → classifier → BIO grouping → summaries; guarded control with fallbacks and timing.
"""
def _merge_bbox(a: List[int], b: List[int]) -> List[int]:
    """Expand a bounding box to include another box (used when merging BIO spans)."""


def _group_sequences(tokens: List[str],
                     bboxes: List[List[int]],
                     labels: List[str],
                     scores: List[float],
                     page_ids: Optional[List[int]] = None,
                     min_conf: float = 0.0) -> List[Dict[str, Any]]:
    """
    Collapse BIO-tagged token predictions into merged field spans. This maps the
    classifier output into "field-level" objects with text, bbox, score, and page.
    """
    groups: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    def push_cur():
        nonlocal cur
        if cur and cur["tokens"]:
            cur["text"] = " ".join(cur["tokens"])
            groups.append(cur)
        cur = None

    for i, lab in enumerate(labels):
        score = scores[i]
        if score < min_conf or lab in ("O", "PAD", None):
            push_cur()
            continue

        prefix, field = (lab.split("-", 1) + [""])[:2] if "-" in lab else ("B", lab)
        tok = tokens[i].strip()
        if not tok:
            continue

        if prefix == "B" or (cur and cur["label"] != field):
            push_cur()
            cur = {
                "label": field,
                "tokens": [tok],
                "bbox": bboxes[i][:],
                "scores": [score],
                "page": page_ids[i] if page_ids else 0
            }
        elif prefix == "I" and cur and cur["label"] == field:
            cur["tokens"].append(tok)
            cur["bbox"] = _merge_bbox(cur["bbox"], bboxes[i])
            cur["scores"].append(score)
        else:
            push_cur()
            cur = {
                "label": field,
                "tokens": [tok],
                "bbox": bboxes[i][:],
                "scores": [score],
                "page": page_ids[i] if page_ids else 0
            }

    push_cur()
    for g in groups:
        g["score"] = float(sum(g.get("scores", [0.0])) / max(1, len(g.get("scores", []))))
        g.pop("scores", None)
    return groups


def _encode(tokens: List[str], bboxes: List[List[int]], max_length: int = 512) -> Dict[str, torch.Tensor]:
    """
    LayoutLMv3 encoding for a single sequence (B=1). Uses a blank canvas to
    satisfy processor requirements while apply_ocr=False; covers the "Fusion
    Layer + Transformer Encoder (Self-Attention)" path in the diagram.
    """
    blank = Image.new("RGB", (1000, 1414), "white")
    return embedding_processor(
        images=[blank],
        text=tokens,
        boxes=bboxes,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )


def analyze_tokens(tokens: List[str],
                   bboxes: List[List[int]],
                   page_ids: Optional[List[int]] = None,
                   config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Skip extraction and run classification + grouping + summarization on tokens.
    This corresponds to the middle and lower sections of the architecture:
      Fusion/Transformer → Graph layer → Dual heads (classifier + decoder).
    """
    cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "min_confidence": 0.0,
        "max_length": 512,
        "graph": {"use": True, "strategy": "knn", "k": 8, "radius": None},
        "model_paths": {"classifier": "saved_models/classifier.pt", "t5": "saved_models/t5.pt"},
        **(config or {}),
    }
    device = torch.device(cfg["device"])

    # Load classifier weights if provided
    if cfg["model_paths"].get("classifier") and os.path.exists(cfg["model_paths"]["classifier"]):
        try:
            sd = torch.load(cfg["model_paths"]["classifier"], map_location="cpu")
            classifier_model.load_state_dict(sd.get("state_dict", sd), strict=False)
        except Exception as e:
            print(f"[doc_pipeline] classifier load skipped ({e})")

    classifier_model.to(device).eval()

    enc = _encode(tokens, bboxes, max_length=cfg["max_length"])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}

    graph = None
    if cfg.get("graph", {}).get("use", False):
        try:
            bboxes_np = batch["bbox"][0].detach().cpu().numpy()
            graph = build_edges(
                bboxes_np,
                strategy=cfg["graph"].get("strategy", "knn"),
                k=int(cfg["graph"].get("k", 8)),
                radius=cfg["graph"].get("radius", None),
                page_ids=None,
            )
        except Exception as e:
            print(f"[doc_pipeline] graph build skipped ({e})")
            graph = None

    t0 = time.time()
    with torch.no_grad():
        out = classifier_model(batch, graph=graph, labels=None)
        logits = out["logits"]
        probs = F.softmax(logits, dim=-1)
        confs, preds = probs.max(dim=-1)
        pred_ids = preds[0].tolist()
        conf_vals = [float(x) for x in confs[0].tolist()]
    classify_ms = int((time.time() - t0) * 1000)

    id2label = getattr(classifier_model, "id2label", None)
    if id2label is None and hasattr(getattr(classifier_model, "backbone", None), "config"):
        id2label = getattr(classifier_model.backbone.config, "id2label", None)
    if id2label is None:
        id2label = {i: f"LABEL_{i}" for i in range(max(pred_ids + [0]) + 1)}

    labels = [id2label.get(i, "O") for i in pred_ids]
    groups = _group_sequences(tokens, bboxes, labels, conf_vals, page_ids=page_ids, min_conf=cfg["min_confidence"])
    fields = summarize_fields(groups, cfg["model_paths"].get("t5"))

    return {
        "document": "tokens",
        "fields": [
            {
                "label": f["label"],
                "score": float(f["score"]),
                "tokens": f["tokens"],
                "bbox": [int(x) for x in f["bbox"]],
                "summary": f.get("summary", ""),
                "page": int(f.get("page", 0)),
            }
            for f in fields
        ],
        "runtime": {"extract_ms": 0, "classify_ms": classify_ms, "summarize_ms": 0},
    }


def analyze_pdf(pdf_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Full stack: extract (pdfplumber) → encode → classify → group → summarize.
    This is the single-call path from uploaded PDF to JSON output used by the UI.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    t0 = time.time()
    try:
        ext = extract_pdf(pdf_path)
        tokens = [str(t.text) for t in ext.tokens]
        bboxes = [list(t.bbox) for t in ext.tokens]
        page_ids = [int(t.page) for t in ext.tokens]
    except Exception as e:
        print(f"[doc_pipeline] extraction failed: {e}")
        tokens, bboxes, page_ids = ["Sample"], [[20, 20, 120, 50]], [0]
    extract_ms = int((time.time() - t0) * 1000)

    result = analyze_tokens(tokens, bboxes, page_ids=page_ids, config=config or {})
    result["document"] = pdf_path
    result["runtime"]["extract_ms"] = extract_ms
    return result


# -----------------------------------------------------------------------------
# 5) Summarization (T5-based decoder)
# -----------------------------------------------------------------------------
"""
Program Title:
  - (4) Graph construction (node/edge builder for GNN)
  
Programmers:
  - Dinglasa, Roanne Maye B.
  - Espartero, Rasty C.
  - Mahayag, David Geisler M.
  - Placente, Yesa V.
  
Where the program fits in the general system designs:
  - Creates spatial neighbor edges feeding the GNN layout layer.
  
Date written: August 2025
Date revised: October 2025

Purpose:
  - Provide a consistent graph representation of spatial relationships between
    tokens so the GNN can propagate information along “nearby” nodes.
  - Support both k-NN and radius-based neighborhood construction while keeping
    edges restricted within the same page when requested.
  - Output graph tensors (`edge_index`, `edge_attr`, `num_nodes`) in a format
    directly consumable by the GNN block in the classifier.

Data structures, algorithms, and control:
  - Data structures:
      - Input `bboxes: np.ndarray[T, 4]` containing normalized token boxes
        (x0, y0, x1, y1) in the 0–1000 LayoutLM coordinate space.
      - Optional `page_ids: np.ndarray[T]` marking which page each token belongs
        to, used to prevent edges from crossing page boundaries.
      - Output dict with:
          - `edge_index: torch.LongTensor[2, E]` listing directed edges
            (source → target node indices).
          - `edge_attr: torch.FloatTensor[E, 3]` storing per-edge features
            `[distance, dx, dy]` for geometry-aware message passing.
          - `num_nodes: int` representing the number of tokens/nodes T.
  - Algorithms:
      - Compute token centers `(cx, cy)` from each bounding box to serve as the
        basis for distance and neighbor calculations.
      - For `"knn"` strategy:
          - Use `scipy.spatial.KDTree` when available for efficient nearest
            neighbor queries; otherwise fall back to a brute-force distance matrix.
          - For each node, query its k nearest neighbors (excluding itself) and
            build edges `(i → j)` with associated distance and delta offsets.
      - For `"radius"` strategy:
          - Compute pairwise distances and connect all nodes within a specified
            radius, again excluding self-edges.
      - Apply `page_ids` filtering (when provided) so only tokens on the same
        page are connected, matching the visual layout assumption of the model.
      - If no edges are produced (e.g., very small T), fall back to a simple
        sequential chain `0→1→2→…` to keep the GNN graph well-defined.
  - Control:
      - Validate the `strategy` argument and raise a `ValueError` for unknown
        strategies to fail fast during configuration errors.
      - Keep graph construction deterministic for a given set of bboxes and
        parameters (no random sampling of neighbors).
      - Return the packed edge tensors in a single dictionary that upstream
        code can pass directly into the GNN-enhanced classifier.
"""

def _load_t5(model_path: Optional[str]) -> Tuple[Optional[AutoTokenizer], Optional[T5ForConditionalGeneration], torch.device]:
    """
    Lazy loader that supports HF directories, hub IDs, or raw .pt state_dicts.
    Defaults to google/flan-t5-base when no path is provided. Mirrors the
    "T5-based Decoder" branch in the diagram.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = "google/flan-t5-base"

    try:
        if model_path is None:
            tok = AutoTokenizer.from_pretrained(base, use_fast=True)
            mdl = T5ForConditionalGeneration.from_pretrained(base)
        elif os.path.isdir(model_path):
            tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            mdl = T5ForConditionalGeneration.from_pretrained(model_path)
        elif os.path.isfile(model_path) and model_path.lower().endswith(".pt"):
            tok = AutoTokenizer.from_pretrained(base, use_fast=True)
            mdl = T5ForConditionalGeneration.from_pretrained(base)
            state = torch.load(model_path, map_location="cpu")
            mdl.load_state_dict(state, strict=False)
        else:
            tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            mdl = T5ForConditionalGeneration.from_pretrained(model_path)

        mdl.to(device).eval()
        return tok, mdl, device
    except Exception:
        return None, None, device


def _template_summary(group: Dict[str, Any]) -> str:
    label = str(group.get("label", "")).replace("_", " ").strip().title()
    text = " ".join(group.get("tokens", [])).strip()
    return f"{label}: {text}" if text else f"{label} field."


def summarize_fields(groups: List[Dict[str, Any]],
                     t5_path: Optional[str] = None,
                     max_new_tokens: int = 32,
                     num_beams: int = 4) -> List[Dict[str, Any]]:
    """
    Generate concise summaries per grouped field span using a T5 encoder-decoder.
    Falls back to template summaries when model download or weights are absent.
    Produces the "class, confidence, summary" triple in the JSON output.
    """
    tok, mdl, device = _load_t5(t5_path)
    outputs: List[Dict[str, Any]] = []

    if tok is None or mdl is None:
        for g in groups:
            g = dict(g)
            g["summary"] = _template_summary(g)
            g.setdefault("score", 0.0)
            outputs.append(g)
        return outputs

    prompts = [
        f"Explain this form field in one short sentence.\nField: {g.get('label','')}\nValue: {' '.join(g.get('tokens', []))}"
        for g in groups
    ]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        gen = mdl.generate(**enc, max_new_tokens=max_new_tokens, num_beams=num_beams)
    dec = tok.batch_decode(gen, skip_special_tokens=True)

    for g, text in zip(groups, dec):
        g = dict(g)
        g["summary"] = text.strip() or _template_summary(g)
        g.setdefault("score", 0.0)
        outputs.append(g)
    return outputs


# -----------------------------------------------------------------------------
# Convenience wrapper to show architecture flow in a single call
# -----------------------------------------------------------------------------
def run_full_pipeline(pdf_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    High-level demo entrypoint matching the architecture diagram:
      PDF -> Extraction -> LayoutLMv3 (text/pos/2D/visual) -> Fusion -> Self-Attn ->
      GNN message passing (edges from spatial neighbors) -> Dual heads (classifier + T5 decoder) ->
      Structured JSON output.

    This is documentation-only and mirrors the production pipeline in one place.
    """
    return analyze_pdf(pdf_path, config=config or {})


# -----------------------------------------------------------------------------
# 6) UI connectors (FastAPI thin layer)
# -----------------------------------------------------------------------------
"""
Program Title:
  - IntelliForm Pipeline (UI Connectors)
Programmers:
  - Dinglasa, Roanne Maye B.
  - Espartero, Rasty C.
  - Mahayag, David Geisler M.
  - Placente, Yesa V.
Where the program fits in the general system designs:
  - Bridges web UI uploads/analyze calls to pipeline and exposes metrics endpoint.
Date written and revised: 2024-09
Purpose:
  - Provide doc-ready FastAPI stubs showing upload/analyze/metrics flow.
Data structures, algorithms, and control:
  - HTTP request bodies → pipeline calls → JSON responses + metrics text file; structured request validation.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body  # type: ignore
from fastapi.responses import JSONResponse, PlainTextResponse  # type: ignore

app = FastAPI(title="IntelliForm API (Docs Extract)", version="doc-only")


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    UI entrypoint: accepts PDF upload, stores under /uploads, computes canonical hash,
    and returns identifiers consumed by the web client for subsequent steps.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    safe_name = file.filename.replace(" ", "_")
    stored = f"demo_{safe_name}"
    disk_path = str(Path("uploads") / stored)
    with open(disk_path, "wb") as f:
        f.write(await file.read())
    template_hash = "HASHED_ID_FOR_DOCS"  # placeholder for canonical hash logic
    return {
        "ok": True,
        "web_path": f"/uploads/{stored}",
        "disk_path": disk_path,
        "canonical_form_id": template_hash,
        "form_id": template_hash,
    }


@app.post("/api/analyze")
async def api_analyze(
    file_path: Optional[str] = Query(default=None, description="Web path (/uploads/...)"),
    body: Optional[Dict[str, Any]] = Body(default=None),
):
    """
    Bridge from UI → model service. Accepts either an uploaded PDF path or a JSON
    payload of tokens/bboxes, then runs analyze_pdf/analyze_tokens and returns
    fields + runtime + a quick metrics stub to the front end.
    """
    cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "min_confidence": (body or {}).get("min_confidence", 0.0) if body else 0.0,
        "max_length": 512,
        "graph": {"use": True, "strategy": "knn", "k": 8, "radius": None},
        "model_paths": {"classifier": "saved_models/classifier.pt", "t5": "saved_models/t5.pt"},
    }

    t0 = time.time()
    if file_path:
        result = analyze_pdf(file_path, config=cfg)
        title = os.path.basename(file_path)
    else:
        if not body:
            raise HTTPException(status_code=400, detail="Provide file_path or tokens/bboxes JSON.")
        result = analyze_tokens(body["tokens"], body["bboxes"], page_ids=body.get("page_ids"), config=cfg)
        title = "tokens.json"

    elapsed = time.time() - t0
    fields = result.get("fields", [])
    runtime = result.get("runtime", {})

    # lightweight text report for UI metrics view
    quick_lines = [
        "IntelliForm — Metrics (Quick Report)",
        f"Field Count: {len(fields)}",
        f"Processing (s): {elapsed:.3f}",
        f"Runtimes (ms): extract={runtime.get('extract_ms','-')} classify={runtime.get('classify_ms','-')} summarize={runtime.get('summarize_ms','-')}",
    ]
    Path("static").mkdir(exist_ok=True)
    Path("static/metrics_report.txt").write_text("\n".join(quick_lines), encoding="utf-8")

    return JSONResponse(content={
        "document": file_path or title,
        "title": title,
        "fields": fields,
        "metrics": {
            "precision": None, "recall": None, "f1": None,
            "rougeL": None, "meteor": None,
            "fields_count": len(fields),
            "processing_sec": elapsed,
        },
        "runtime": runtime,
    })


@app.get("/api/metrics", response_class=PlainTextResponse)
def api_metrics():
    """
    Serves the current metrics text report (written after /api/analyze or by the reporter).
    """
    path = Path("static/metrics_report.txt")
    if not path.exists():
        return PlainTextResponse("No metrics report found.", status_code=404)
    return PlainTextResponse(path.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# 7) Metrics logging and computation (training/inference parity)
# -----------------------------------------------------------------------------
"""
Program Title:
  - IntelliForm Pipeline (Metrics Computation)
Programmers:
  - Dinglasa, Roanne Maye B.
  - Espartero, Rasty C.
  - Mahayag, David Geisler M.
  - Placente, Yesa V.
Where the program fits in the general system designs:
  - Computes and formats core evaluation metrics for classification, spans, and summaries.
Date written and revised: 2024-09
Purpose:
  - Supply PRF, IoU spans, ROUGE/METEOR scaffolds, and text report rendering.
Data structures, algorithms, and control:
  - numpy arrays → PRF; IoU pairing; simple ROUGE/METEOR placeholders; formatted text report.
"""
def compute_prf(y_true: List[int], y_pred: List[int], mask: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Token-level precision/recall/F1 (micro). Mirrors utils.metrics.compute_prf.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        yt, yp = yt[m], yp[m]
    if yt.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    TP = (yt == yp).sum()
    acc = float(TP) / float(yt.size)
    return {"precision": acc, "recall": acc, "f1": acc}


def evaluate_spans_iou(predicted: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]], iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Field-level IoU matching for span detection. Produces TP/FP/FN plus P/R/F1.
    """
    def _iou(boxA: List[int], boxB: List[int]) -> float:
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter <= 0:
            return 0.0
        areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        denom = areaA + areaB - inter
        return float(inter) / float(denom) if denom > 0 else 0.0

    TP = FP = 0
    used = set()
    for p in predicted:
        p_box, p_lbl = p.get("bbox"), p.get("label")
        if p_box is None or p_lbl is None:
            FP += 1
            continue
        best_iou, best_j = 0.0, None
        for j, g in enumerate(ground_truth):
            if j in used or p_lbl != g.get("label"):
                continue
            giou = _iou(p_box, g.get("bbox"))
            if giou > best_iou:
                best_iou, best_j = giou, j
        if best_iou >= iou_threshold and best_j is not None:
            TP += 1
            used.add(best_j)
        else:
            FP += 1
    FN = len(ground_truth) - len(used)
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"TP": TP, "FP": FP, "FN": FN, "precision": precision, "recall": recall, "f1": f1}


def compute_rouge_meteor(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    ROUGE-L and METEOR over lists of strings (graceful fallback if deps absent).
    """
    n = min(len(references), len(hypotheses))
    if n == 0:
        return {"rougeL": 0.0, "meteor": 0.0}
    rougeL = float(np.mean([1.0 if r == h else 0.0 for r, h in zip(references[:n], hypotheses[:n])]))
    meteor = rougeL
    return {"rougeL": rougeL, "meteor": meteor}


def format_metrics_for_report(classif: Optional[Dict[str, float]] = None,
                              summar: Optional[Dict[str, float]] = None,
                              spans: Optional[Dict[str, float]] = None,
                              header: str = "IntelliForm — Metrics Report") -> str:
    """
    Render a text report (used by metrics endpoint and research dashboard).
    """
    lines: List[str] = [header, "=" * len(header), ""]
    if classif:
        lines += [
            "Token Classification:",
            f"  Precision : {classif.get('precision',0.0):.4f}",
            f"  Recall    : {classif.get('recall',0.0):.4f}",
            f"  F1        : {classif.get('f1',0.0):.4f}",
            "",
        ]
    if spans:
        lines += [
            "Field Spans (IoU):",
            f"  TP/FP/FN : {int(spans.get('TP',0))}/{int(spans.get('FP',0))}/{int(spans.get('FN',0))}",
            f"  Precision: {spans.get('precision',0.0):.4f}",
            f"  Recall   : {spans.get('recall',0.0):.4f}",
            f"  F1       : {spans.get('f1',0.0):.4f}",
            "",
        ]
    if summar:
        lines += [
            "Summary Generation:",
            f"  ROUGE-L  : {summar.get('rougeL',0.0):.4f}",
            f"  METEOR   : {summar.get('meteor',0.0):.4f}",
            "",
        ]
    return "\n".join(lines).strip() + "\n"


def save_report_txt(text: str, path: str = "static/metrics_report.txt") -> None:
    """Persist the rendered metrics report for UI retrieval (/api/metrics)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# -----------------------------------------------------------------------------
# 8) Metrics logging and gentle post-processing (UI dashboards)
# -----------------------------------------------------------------------------
"""
Program Title:
  - IntelliForm Pipeline (Metrics Logging)
Programmers:
  - Dinglasa, Roanne Maye B.
  - Espartero, Rasty C.
  - Mahayag, David Geisler M.
  - Placente, Yesa V.
Where the program fits in the general system designs:
  - Writes stabilized metrics/user rows for researcher dashboards and undo/delete ops.
Date written and revised: 2024-09
Purpose:
  - Append tool/user metrics with gentle clamping and deterministic IDs for UI consumption.
Data structures, algorithms, and control:
  - metrics dicts → clamped/rounded values → JSONL rows with stable hash IDs; simple append control flow.
"""
import hashlib

LOGS_DIR = Path("explanations") / "logs"
TOOL_LOG = LOGS_DIR / "tool-metrics.jsonl"
USER_LOG = LOGS_DIR / "user-metrics.jsonl"


def _stable_row_id(row: Dict[str, Any]) -> str:
    """Deterministic hash per metrics row so UI can delete/undo entries reliably."""
    payload = dict(row)
    payload.pop("row_id", None)
    s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def append_tool_metrics(row: Dict[str, Any]) -> bool:
    """
    Log analyzer metrics for a form (precision/recall/F1, rougeL/meteor, timing).
    Tiny stabilizing tweaks are applied (clip to [0,1], round to 3 decimals) to
    reflect the final reported values after computation.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = dict(row.get("metrics") or {})
    for k, v in list(metrics.items()):
        try:
            metrics[k] = round(min(max(float(v), 0.0), 1.0), 3)
        except Exception:
            metrics[k] = 0.0
    row = dict(row)
    row["metrics"] = metrics
    row["row_id"] = row.get("row_id") or _stable_row_id(row)
    row["ts"] = row.get("ts") or int(time.time() * 1000)
    try:
        with open(TOOL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def append_user_metrics(row: Dict[str, Any]) -> bool:
    """Log user-facing completion-time and method comparisons (IntelliForm vs manual)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row["row_id"] = row.get("row_id") or _stable_row_id(row)
    row["ts"] = row.get("ts") or int(time.time() * 1000)
    try:
        with open(USER_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def tweak_metrics(canonical_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gentle normalization pass used before logging to keep reports stable:
      - clamp metrics to [0,1]
      - round to 3 decimals
      - copy through TP/FP/FN if present (integerized)
    Represents the post-compute adjustment before dashboards read the values.
    """
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        if k.lower() in {"tp", "fp", "fn"}:
            try:
                out[k.lower()] = int(v)
                continue
            except Exception:
                out[k.lower()] = 0
                continue
        try:
            out[k] = round(min(max(float(v), 0.0), 1.0), 3)
        except Exception:
            out[k] = 0.0
    out.setdefault("canonical_id", canonical_id)
    return out


__all__ = [
    "extract_pdf",
    "build_edges",
    "FieldClassifier",
    "analyze_tokens",
    "analyze_pdf",
    "summarize_fields",
    "run_full_pipeline",
    "app",
    "api_upload",
    "api_analyze",
    "api_metrics",
    "compute_prf",
    "evaluate_spans_iou",
    "compute_rouge_meteor",
    "format_metrics_for_report",
    "save_report_txt",
    "append_tool_metrics",
    "append_user_metrics",
    "tweak_metrics",
]
