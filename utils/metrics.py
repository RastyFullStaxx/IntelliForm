"""
IntelliForm — Metrics (Classification + Generation)
===================================================

WHAT THIS MODULE DOES
---------------------
Collects and computes evaluation metrics for:
- Token-level classification (Precision, Recall, F1 — micro/macro support).
- Summary generation (ROUGE-L, METEOR) for T5 outputs.

WHEN IT'S USED
--------------
- **Training**: called each epoch to log validation metrics and early-stopping signals.
- **Inference/Evaluation**: used to produce final `metrics_report.txt`.

INPUTS
------
Classification:
  - y_true: List[int] or 1D Tensor of true labels (flattened over tokens, ignoring pad)
  - y_pred: same shape, predicted labels
  - label_mask: optional mask to exclude special tokens/padding

Generation:
  - references: List[str] (ground truth summaries)
  - hypotheses: List[str] (model-generated summaries)

OUTPUTS
-------
- dicts with:
  - "precision", "recall", "f1" (micro/macro/weighted optional)
  - "rougeL", "meteor" for generation
- pretty_printer to format a table for logs / `metrics_report.txt`

KEY FUNCTIONS
-------------
- compute_prf(y_true, y_pred, label_mask=None, average="micro") -> dict
- compute_rouge_meteor(references, hypotheses) -> dict
- update_running_metrics(running, batch_metrics) -> dict
- format_metrics_for_report(classif: dict, summar: dict) -> str

DEPENDENCIES
------------
- numpy
- sklearn.metrics (precision_recall_fscore_support) or custom
- nltk (METEOR) and/or `rouge-score` / `datasets` for ROUGE-L

INTERACTIONS
------------
- Called by: scripts/train_classifier.py, scripts/train_t5.py, utils/llmv3_infer.py
- Outputs written to: metrics logs + `metrics_report.txt`

EXTENSION POINTS / TODOs
------------------------
- Add per-label F1 breakdown for confusion hotspots.
- Add ECE calibration (Expected Calibration Error) if you want confidence reporting.

"""


import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import single_meteor_score


def calculate_iou(boxA, boxB):
    # box format: [x0, y0, x1, y1]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def evaluate_field_predictions(predicted, ground_truth, iou_threshold=0.5):
    TP, FP, FN = 0, 0, 0
    matched_gt = set()

    for pred in predicted:
        pred_box = pred['box']
        pred_label = pred['label']

        best_iou = 0
        best_gt = None

        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            if pred_label != gt['label']:
                continue

            iou = calculate_iou(pred_box, gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_iou >= iou_threshold:
            TP += 1
            matched_gt.add(best_gt)
        else:
            FP += 1

    FN = len(ground_truth) - len(matched_gt)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }


def evaluate_summaries(pred_summaries, gt_summaries):
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    rouge_scores = []
    meteor_scores = []

    for pred, gt in zip(pred_summaries, gt_summaries):
        rougeL = rouge.score(gt, pred)['rougeL'].fmeasure
        meteor = single_meteor_score(gt, pred)
        rouge_scores.append(rougeL)
        meteor_scores.append(meteor)

    avg_rougeL = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    return {
        'ROUGE-L': avg_rougeL,
        'METEOR': avg_meteor
    }
    
    def compute_dummy_ece_score(predicted_confidences, predicted_labels, true_labels, num_bins=10):
    """
    Dummy ECE computation using fixed bins.
    predicted_confidences: list of float (confidence scores from model)
    predicted_labels: list of predicted class labels
    true_labels: list of ground truth class labels
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = [j for j, conf in enumerate(predicted_confidences) if bin_lower < conf <= bin_upper]

        if len(in_bin) > 0:
            acc = np.mean([predicted_labels[j] == true_labels[j] for j in in_bin])
            conf_avg = np.mean([predicted_confidences[j] for j in in_bin])
            ece += (len(in_bin) / len(predicted_confidences)) * abs(acc - conf_avg)

    return ece


    def save_analysis_to_txt(metrics_dict, file_path='static/metrics_report.txt'):
        """
        Saves any dictionary of metrics to a .txt file
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for key, value in metrics_dict.items():
                f.write(f"{key}: {value:.4f}\n")
