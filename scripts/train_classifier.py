"""
IntelliForm — Classifier Training Pipeline
==========================================

WHAT THIS SCRIPT DOES
---------------------
Trains the LayoutLMv3 + GNN + classifier model to perform token-level
field label classification on annotated PDF form datasets (e.g., XFUND, FUNSD).

It:
1. Loads tokenized + annotated datasets from `data/train/annotations/` and `data/val/annotations/`
   via `utils.dataset_loader`.
2. Builds graph edges for each sample via `utils.graph_builder` if GNN is enabled.
3. Instantiates `utils.field_classifier.FieldClassifier` with the desired config.
4. Runs the training loop (forward, loss, backward, optimizer step) with PyTorch.
5. Logs training/validation metrics (Precision, Recall, F1) via `utils.metrics`.
6. Saves the trained model checkpoint to `saved_models/classifier.pt`.

PRIMARY INPUTS
--------------
- Config (YAML/JSON/dict or argparse CLI) specifying:
    * model_name_or_path (LayoutLMv3 variant)
    * num_labels, label_map_path
    * use_gnn, gnn_hidden_dim, edge_strategy, k/radius
    * optimizer & scheduler settings
    * epochs, batch_size, device
- Training/validation data in model-ready JSON format (tokens, bboxes, labels).

OUTPUTS
-------
- Trained model weights: `saved_models/classifier.pt`
- Training log file: `metrics/log_classifier.txt`
- Optional: intermediate checkpoints in `checkpoints/` if enabled.

KEY STEPS
---------
1. Parse config/args, load label map from `labels.json`.
2. Initialize tokenizer (LayoutLMv3TokenizerFast) and dataset loaders.
3. Initialize FieldClassifier model (optionally with pre-trained weights).
4. Prepare optimizer (AdamW) and scheduler (linear warmup/decay).
5. Train for N epochs:
    - forward pass -> loss
    - backward pass -> optimizer step
    - periodic validation -> metrics
6. Save final and/or best checkpoint.

DEPENDENCIES
------------
- utils.dataset_loader
- utils.field_classifier
- utils.graph_builder
- utils.metrics
- torch, transformers

INTERACTIONS
------------
- Consumes: `data/train/annotations`, `data/val/annotations`
- Produces: `saved_models/classifier.pt`
- Used by: `scripts/train_all.py` for end-to-end training.

EXTENSION POINTS / TODOs
------------------------
- Add mixed-precision training (fp16) for speed/memory savings.
- Add per-label breakdown in validation reports.
- Integrate early stopping on validation F1.

"""
