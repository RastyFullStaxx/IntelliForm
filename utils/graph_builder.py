"""
IntelliForm — Graph Builder for GNN
===================================

WHAT THIS MODULE DOES
---------------------
Constructs graph connectivity (edges) over document tokens to let the GNN
propagate information across spatially/semantically related tokens.

TYPICAL EDGE STRATEGIES
-----------------------
- k-NN in 2D layout space (using bbox centers or token centroids).
- Radius-based neighbors (within a pixel/layout distance threshold).
- Row/column heuristics to link visually aligned tokens.
- Optional: lexical cues (e.g., colon, key-value patterns) to add edges.

WHEN IT'S USED
--------------
- **Training & Inference**: called upstream (dataset loader or collate_fn) to compute
  per-sample graph edges fed into `utils.field_classifier.FieldClassifier`.

PRIMARY INPUTS
--------------
- tokens: List[str] or token ids (optional)
- bboxes: np.ndarray or Tensor [T, 4] (x0, y0, x1, y1) in normalized coords
- (optional) page ids if multi-page
- config: dict controlling k, radius, max_edges, alignment tolerances, etc.

OUTPUTS
-------
- graph dict with:
    edge_index: LongTensor [2, E]
    edge_attr:  FloatTensor [E, D] (optional, e.g., dx, dy, distance, same_row flag)
    num_nodes:  int (T)
  Designed to be fed into a simple GNN layer inside `field_classifier.py`.

KEY FUNCTIONS
-------------
- build_edges(bboxes: np.ndarray, strategy: str = "knn", k: int = 8, **kwargs) -> dict
- build_features(bboxes: np.ndarray, edges: np.ndarray, **kwargs) -> np.ndarray
- pack_graph(edge_index, edge_attr, num_nodes) -> dict

DEPENDENCIES
------------
- numpy, torch
- (optional) scipy.spatial for k-NN; otherwise implement simple k-NN

INTERACTIONS
------------
- Called by: dataset loader (during batch creation) or directly by training/inference
- Consumed by: utils.field_classifier (GNN forward pass)

EXTENSION POINTS / TODOs
------------------------
- Add page-break aware edges.
- Add semantic edges from predicted BIO tags (2-pass refinement).
- Cache edges to speed up repeated runs.

"""
