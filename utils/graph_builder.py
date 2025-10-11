# utils/graph_builder.py

"""
IntelliForm Graph Builder for GNN
=================================

WHAT THIS MODULE DOES
---------------------
Constructs graph connectivity (edges) over document tokens so that the GNN
can propagate contextual information across spatially or visually related tokens.

TYPICAL EDGE STRATEGIES
-----------------------
- **k-NN** in 2D layout space (using bbox centers).
- **Radius-based** neighbors (all tokens within distance threshold).
- **Row/Column heuristics**: link visually aligned tokens.
- **Optional lexical cues**: e.g., colon, key-value patterns.

WHEN IT'S USED
--------------
- **Training & Inference**: called upstream (dataset loader or collate_fn) to compute
  per-sample graph edges fed into `utils.field_classifier.FieldClassifier`.

PRIMARY INPUTS
--------------
- bboxes : np.ndarray or torch.Tensor [T, 4]
           (x0, y0, x1, y1) in normalized coords (0–1000).
- strategy : str, one of {"knn", "radius"}.
- k : int (for knn), number of neighbors.
- radius : float (for radius), max distance in bbox-centroid space.
- page_ids : optional array to restrict edges within the same page.

OUTPUTS
-------
A dict with:
- edge_index : torch.LongTensor [2, E]
- edge_attr  : torch.FloatTensor [E, D] (e.g., dx, dy, distance, same_row flag)
- num_nodes  : int (T)

KEY FUNCTIONS
-------------
- build_edges(bboxes, strategy="knn", k=8, radius=None, page_ids=None) -> dict
- build_features(bboxes, edges) -> np.ndarray
- pack_graph(edge_index, edge_attr, num_nodes) -> dict

DEPENDENCIES
------------
- numpy, torch
- scipy.spatial.KDTree (optional, for k-NN acceleration)

INTERACTIONS
------------
- Called by: dataset loader (batch creation) or directly in training/inference.
- Consumed by: utils.field_classifier (GNN forward pass).

EXTENSION POINTS / TODOs
------------------------
- Add page-break-aware edges (prevent cross-page connections).
- Add semantic edges from BIO predictions (2-pass refinement).
- Cache edges to speed up repeated runs.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Tuple, Optional

try:
    from scipy.spatial import KDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _centers(bboxes: np.ndarray) -> np.ndarray:
    """Compute centers (x,y) from [x0,y0,x1,y1] boxes."""
    return np.stack([(bboxes[:, 0] + bboxes[:, 2]) / 2.0,
                     (bboxes[:, 1] + bboxes[:, 3]) / 2.0], axis=1)


def build_edges(
    bboxes: np.ndarray,
    strategy: str = "knn",
    k: int = 8,
    radius: Optional[float] = None,
    page_ids: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build graph edges given token bounding boxes.

    Args:
        bboxes   : np.ndarray [T,4] of token boxes (normalized 0–1000).
        strategy : "knn" or "radius".
        k        : number of neighbors (for knn).
        radius   : distance threshold (for radius).
        page_ids : np.ndarray [T], restricts edges within same page if provided.

    Returns:
        dict with edge_index, edge_attr, num_nodes.
    """
    num_nodes = bboxes.shape[0]
    centers = _centers(bboxes)
    edge_list = []

    if strategy == "knn":
        if _HAS_SCIPY:
            tree = KDTree(centers)
            for i, c in enumerate(centers):
                # query k+1 because first neighbor is itself
                dists, idxs = tree.query(c, k=k + 1)
                for j, d in zip(idxs[1:], dists[1:]):  # skip self
                    if page_ids is not None and page_ids[i] != page_ids[j]:
                        continue
                    edge_list.append((i, j, d, centers[j][0] - c[0], centers[j][1] - c[1]))
        else:
            # fallback: brute-force
            dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            for i in range(num_nodes):
                idxs = np.argsort(dmat[i])[:k + 1]
                for j in idxs:
                    if i == j:
                        continue
                    if page_ids is not None and page_ids[i] != page_ids[j]:
                        continue
                    d = dmat[i, j]
                    edge_list.append((i, j, d, centers[j][0] - centers[i][0],
                                      centers[j][1] - centers[i][1]))

    elif strategy == "radius":
        assert radius is not None, "radius must be set for radius strategy"
        dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        for i in range(num_nodes):
            neighbors = np.where(dmat[i] <= radius)[0]
            for j in neighbors:
                if i == j:
                    continue
                if page_ids is not None and page_ids[i] != page_ids[j]:
                    continue
                d = dmat[i, j]
                edge_list.append((i, j, d, centers[j][0] - centers[i][0],
                                  centers[j][1] - centers[i][1]))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if not edge_list:
        # fallback: connect sequentially
        edge_list = [(i, i + 1, 1.0, 1.0, 0.0) for i in range(num_nodes - 1)]

    edge_index = torch.tensor([[i, j] for (i, j, *_)
                               in edge_list], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([[d, dx, dy] for (_, _, d, dx, dy)
                              in edge_list], dtype=torch.float)

    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": num_nodes,
    }


def pack_graph(edge_index: torch.Tensor,
               edge_attr: torch.Tensor,
               num_nodes: int) -> Dict[str, torch.Tensor]:
    """Convenience wrapper to pack graph tensors into dict format."""
    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": num_nodes,
    }
