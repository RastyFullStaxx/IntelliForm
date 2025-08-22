# utils/field_classifier.py

"""
IntelliForm — LayoutLMv3 + GNN + Classifier
===========================================

WHAT THIS MODULE DOES
---------------------
Implements the IntelliForm backbone:
- LayoutLMv3 encoder for token-level representations
- Optional lightweight GNN to propagate context over layout edges
- Linear head projecting to label logits

WHEN IT'S USED
--------------
- Training: scripts/train_classifier.py
- Inference: utils/llmv3_infer.py

PRIMARY INPUTS
--------------
Batch dict (LayoutLM-style):
  input_ids      : LongTensor [B, T]
  attention_mask : LongTensor [B, T]
  bbox           : LongTensor [B, T, 4]
  (optional) pixel_values: FloatTensor [B, 3, H, W] if using image branch
  (optional) labels      : LongTensor [B, T] for loss computation

Graph dict (optional, per-sample or per-batch):
  edge_index : LongTensor [2, E]  (node indices in [0..T-1] for each sample)
  edge_attr  : FloatTensor [E, D] (optional geometric features)
  num_nodes  : int (== T)

OUTPUTS
-------
forward(...) -> dict with:
  logits : FloatTensor [B, T, num_labels]
  loss   : scalar (if labels provided)
  hidden : FloatTensor [B, T, H] (optional introspection)

KEY CLASSES / FUNCTIONS
-----------------------
- FieldClassifier(nn.Module): LayoutLMv3 (+ optional GNN) + classifier head
- predict(...): no_grad inference helper returning ids & scores

DEPENDENCIES
------------
- torch, torch.nn.functional
- transformers (LayoutLMv3Model, LayoutLMv3Processor)

INTERACTIONS
------------
- Called by: train scripts & llmv3_infer
- Consumes graph from: utils.graph_builder (optional)

NOTES
-----
- We default to text+layout only (no OCR). Set processor with apply_ocr=False.
- This GNN is a minimal scatter-based layer (no external PyG dependency).
- For batched graphs, pass a list[graph] of length B or None.
"""

from __future__ import annotations
from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMv3Model, LayoutLMv3Processor


# ------------------------------
# Minimal scatter util (no PyG)
# ------------------------------
def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
    out = src.new_zeros(dim_size, *src.shape[1:])
    return out.index_add(dim, index, src)


# ------------------------------
# Simple GNN Layer
# ------------------------------
class SimpleGNNLayer(nn.Module):
    """
    A light message-passing layer:
      m_ij = phi([h_j, e_ij])       (edge-conditioned message)
      h_i' = LayerNorm(h_i + agg_j m_ij)  (residual + norm)

    Shapes:
      H: [T, D], edge_index: [2, E], edge_attr: [E, D_e] or None
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
        # H: [T, D]; edge_index: [2,E]
        src, dst = edge_index[0], edge_index[1]  # j -> i (message from src=j to dst=i)

        if edge_attr is None:
            edge_attr = torch.zeros((src.size(0), 0), device=H.device, dtype=H.dtype)

        Hj = H[src]                              # [E, D]
        feat = torch.cat([Hj, edge_attr], dim=-1) if edge_attr.numel() else Hj
        m = self.phi(feat)                       # [E, D]

        # Aggregate messages onto destination nodes
        T, D = H.size(0), H.size(1)
        agg = scatter_add(m, dst, dim=0, dim_size=T)  # [T, D]

        H_out = self.norm(H + agg)
        return H_out


class GNNBlock(nn.Module):
    """Stack of SimpleGNNLayer(s)."""
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


# ------------------------------
# FieldClassifier
# ------------------------------
class FieldClassifier(nn.Module):
    def __init__(
        self,
        num_labels: int,
        backbone_name: str = "microsoft/layoutlmv3-base",
        use_gnn: bool = True,
        gnn_layers: int = 1,
        edge_dim: int = 3,   # matches graph_builder edge_attr=[dist, dx, dy]
        dropout: float = 0.1,
        use_image_branch: bool = False,
    ):
        super().__init__()
        self.backbone = LayoutLMv3Model.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size

        self.use_gnn = use_gnn
        self.gnn = GNNBlock(hidden_dim=hidden, edge_dim=edge_dim, num_layers=gnn_layers) if use_gnn else None

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

        # For optional image branch (we keep the flag; processor controls pixels)
        self.use_image_branch = use_image_branch

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        graph: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          batch : dict with input_ids, attention_mask, bbox, (optional) pixel_values
          graph : dict or list[dict]; if list, must align with B and we iterate per sample
          labels: [B, T] for loss

        Returns:
          dict: {"logits", "loss"(opt), "hidden"}
        """
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "bbox": batch["bbox"],
        }
        if self.use_image_branch and "pixel_values" in batch:
            inputs["pixel_values"] = batch["pixel_values"]

        out = self.backbone(**inputs)
        H = out.last_hidden_state  # [B, T, D]

        # Apply GNN per-sample (keeps graph API simple without PyG batching)
        if self.use_gnn and graph is not None:
            if isinstance(graph, list):
                # graph is a list of length B; loop over batch items
                H_list = []
                for b in range(H.size(0)):
                    Hb = H[b]  # [T, D]
                    gb = graph[b] if b < len(graph) else None
                    H_list.append(self.gnn(Hb, gb))
                H = torch.stack(H_list, dim=0)
            else:
                # assume B==1
                H = self.gnn(H.squeeze(0), graph).unsqueeze(0)

        H = self.dropout(H)
        logits = self.classifier(H)  # [B, T, num_labels]

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
        """
        Convenience loader to mirror HF style when restoring a whole classifier.
        Expects a state_dict saved by the training script.
        """
        model = cls(num_labels=num_labels, **kwargs)
        sd = torch.load(model_dir_or_name, map_location="cpu")
        # Allow loading either a full dict {"state_dict": ...} or raw state_dict
        state_dict = sd.get("state_dict", sd)
        model.load_state_dict(state_dict, strict=False)
        return model


@torch.no_grad()
def predict(
    model: FieldClassifier,
    batch: Dict[str, torch.Tensor],
    graph: Optional[Dict[str, torch.Tensor]] = None,
    id_to_label: Optional[Dict[int, str]] = None,
    return_hidden: bool = False,
):
    model.eval()
    out = model(batch, graph=graph, labels=None)
    logits = out["logits"]  # [B, T, C]
    probs = F.softmax(logits, dim=-1)
    confs, preds = probs.max(dim=-1)  # [B, T]

    results: List[List[Dict]] = []
    B, T = preds.shape
    for b in range(B):
        seq = []
        for t in range(T):
            label_id = preds[b, t].item()
            if id_to_label:
                label = id_to_label.get(label_id, str(label_id))
            else:
                label = str(label_id)
            seq.append({
                "label_id": label_id,
                "label": label,
                "score": float(confs[b, t].item()),
                "bbox": batch["bbox"][b, t].tolist(),
                "token_id": int(batch["input_ids"][b, t].item()),
            })
        results.append(seq)

    if return_hidden:
        return results, out["hidden"]
    return results


# ------------------------------
# Global processor & models (names expected by your stack)
# ------------------------------
# Primary tokenizer/processor (apply_ocr=False as per project rule)
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False
)

# Base embedding encoder exposed for other modules
embedding = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")

# Full classifier model (initialize with placeholder num_labels; training script should re-init or load)
# You can override num_labels at load-time or reassign `model` after training.
model = FieldClassifier(num_labels=8, use_gnn=True, gnn_layers=1)
