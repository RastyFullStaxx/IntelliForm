"""
IntelliForm — One-Click Full Training Script
============================================

WHAT THIS SCRIPT DOES
---------------------
Convenience script to sequentially train both core IntelliForm components:
1) LayoutLMv3 + GNN + Classifier (`scripts/train_classifier.py`)
2) T5 Summarizer (`scripts/train_t5.py`)

This ensures:
- Both models are trained on matching dataset versions.
- End-to-end IntelliForm inference can be run immediately after.

PRIMARY INPUTS
--------------
- Same configs/arguments as `train_classifier.py` and `train_t5.py`,
  either passed via a master config file or CLI arguments.

OUTPUTS
-------
- `saved_models/classifier.pt` — classifier model weights
- `saved_models/t5.pt` — summarizer model weights
- All logs and metrics from both training runs

KEY STEPS
---------
1. Parse master config/args and resolve paths for both training stages.
2. Call `train_classifier.py` (subprocess or import + function call).
3. Call `train_t5.py` after classifier training completes.
4. Optionally merge logs into a single training report.
5. Print summary of both models' metrics.

DEPENDENCIES
------------
- scripts.train_classifier
- scripts.train_t5
- python subprocess (if calling scripts directly)
- shared configs

INTERACTIONS
------------
- Calls: `train_classifier.py`, `train_t5.py`
- Produces: final trained models + logs
- Often run before model packaging/deployment.

EXTENSION POINTS / TODOs
------------------------
- Add parallel training (if on separate GPUs).
- Add unified WandB/TensorBoard tracking across both stages.
- Allow selective re-training of only one component.

"""
