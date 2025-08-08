"""
IntelliForm — LayoutLMv3 + GNN + Classifier
===========================================

WHAT THIS MODULE DOES
---------------------
Implements the document understanding backbone for IntelliForm:
- Fine-tunes a LayoutLMv3 encoder on token-level field-label classification.
- Optionally applies a Graph Neural Network (GNN) layer to enrich token embeddings
  using spatial/structural edges (from utils.graph_builder).
- Projects enriched embeddings to label logits via a classifier head.

WHEN IT'S USED
--------------
- **Training**: called by `scripts/train_classifier.py` to fit LayoutLMv3 (+ GNN) on
  XFUND/FUNSD-style annotations (tokens, bboxes, labels).
- **Inference**: used by `utils/llmv3_infer.py` to load the trained weights and
  produce field-label predictions with confidences and aligned bboxes.

PRIMARY INPUTS
--------------
- tokenized batch dict with keys (typical LayoutLMv3 format):
    input_ids: LongTensor [B, T]
    attention_mask: LongTensor [B, T]
    bbox: LongTensor [B, T, 4]  (normalized 0..1000 or 0..1 depending on tokenizer config)
    image: (optional) pixel values if using image patches (OCR-heavy docs)
    labels: LongTensor [B, T]  (only at training time)
- graph_edges (optional): from `utils.graph_builder.build_edges()`
    edge_index: LongTensor [2, E]
    edge_attr:  FloatTensor [E, D] (optional)

OUTPUTS
-------
Training forward():
    - loss (CrossEntropyLoss over tokens)
    - logits: FloatTensor [B, T, num_labels]
    - (optional) hidden states for analysis
Inference predict():
    - per-token label ids, confidences
    - token-level metadata (text span, bbox) for post-processing

KEY CLASSES / FUNCTIONS
-----------------------
- class FieldClassifier(nn.Module):
    __init__(..., num_labels: int, use_gnn: bool = True, gnn_hidden: int = 256, ...)
        Loads LayoutLMv3 backbone; attaches optional GNN; adds linear head.

    forward(batch, graph=None, labels=None) -> dict
        Returns {"loss", "logits", "hidden"} as applicable.

    from_pretrained(model_dir_or_name, **kwargs) -> "FieldClassifier"
        Convenience loader that mirrors Hugging Face semantics.

- def predict(model, batch, graph=None, id_to_label=None, threshold=0.0) -> List[dict]
    Runs a no_grad pass and returns structured results:
    [
      {"label": "FULL_NAME", "score": 0.94, "token_idx": 17, "bbox": [x0,y0,x1,y1]},
      ...
    ]

DEPENDENCIES
------------
- transformers: LayoutLMv3Model / LayoutLMv3ForTokenClassification
- torch, torch.nn.functional
- utils.graph_builder (optional): to build graph edges for the GNN
- utils.metrics (only during training if you compute running metrics inside)

INTERACTIONS
------------
- Called by: scripts/train_classifier.py (training), utils/llmv3_infer.py (inference)
- Calls to: utils.graph_builder (optional), torch/transformers internals

TRAINING NOTES
--------------
- Loss: CrossEntropyLoss over token labels (ignore_index for padding/special tokens).
- Optimizer: AdamW; Scheduler: linear warmup + decay (configured in train script).
- Checkpointing handled by train script; model exposes state_dict().

INFERENCE NOTES
---------------
- Ensure identical tokenizer + bbox normalization as training.
- For stable results, keep `id_to_label` consistent with `labels.json`.

EXTENSION POINTS / TODOs
------------------------
- Add CRF decoding for BIO labels if needed.
- Add masking strategy for irregular OCR tokens.
- Optional: export ONNX for accelerated inference.

"""

# utils/field_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMv3Model, LayoutLMv3Processor

# 📌 Define label set
LABELS = [
    "O",
    "B-TIN", "I-TIN",
    "B-DateOfBirth", "I-DateOfBirth",
    "B-Name", "I-Name",
    "B-Address", "I-Address",
    "B-Email", "I-Email",
    "B-ContactNumber", "I-ContactNumber",
    "B-MonthlyIncome", "I-MonthlyIncome",
    "B-Employer", "I-Employer"
]

label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for label, idx in label2id.items()}

# 🧠 Custom classifier with LayoutLMv3 base
class LayoutLMv3FieldClassifier(nn.Module):
    def __init__(self, num_labels=len(LABELS)):
        super(LayoutLMv3FieldClassifier, self).__init__()
        self.encoder = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, bbox, pixel_values):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        return logits

# ⚙️ Load processor and models
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False  # ✅ Prevent re-OCRing; we control bbox
)

model = LayoutLMv3FieldClassifier()
model.eval()

# Raw encoder for embedding use elsewhere
embedding = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")

@torch.no_grad()
def predict_fields(model_inputs):
    print("🧠 Running model forward pass...")

    # ⛑️ Ensure only 1 page worth of pixel_values (shape [1, 3, H, W])
    pixel_values = model_inputs["pixel_values"]
    if pixel_values.dim() == 5:
        pixel_values = pixel_values[:, 0, :, :, :]  # remove batch if nested
    if pixel_values.size(0) > 1:
        pixel_values = pixel_values[0].unsqueeze(0)

    logits = model(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        bbox=model_inputs["bbox"].long(),
        pixel_values=pixel_values
    )

    print("✅ Model forward pass complete.")

    probs = F.softmax(logits, dim=-1)
    confs, preds = torch.max(probs, dim=-1)

    print("📊 Softmax and predictions extracted.")

    results = []

    input_ids = model_inputs["input_ids"]
    bboxes = model_inputs["bbox"][0]  # assuming batch_size=1

    sequence_length = input_ids.size(1)

    for i in range(sequence_length):
        label_id = preds[0][i].item()
        label = id2label[label_id]
        confidence = confs[0][i].item()
        token_id = input_ids[0][i].item()
        text = processor.tokenizer.decode([token_id])

        if label != "O" and text.strip():
            results.append({
                "label": label,
                "bbox": [int(x) for x in bboxes[i]],
                "confidence": round(confidence, 3),
                "text": text
            })

    print("\n🧾 First 10 decoded tokens with predictions:")
    for i in range(min(10, sequence_length)):
        label_id = preds[0][i].item()
        label = id2label[label_id]
        confidence = confs[0][i].item()
        token_id = input_ids[0][i].item()
        text = processor.tokenizer.decode([token_id])
        box = bboxes[i]

        print(f"{text} | Box: {box} | Label: {label} | Confidence: {round(confidence, 3)}")

    return results
