"""
metrics.py

Canonical metric implementations for evaluating segmentation performance.
Reference: Subsections 3.9.2 and 3.9.3 (Evaluation Framework and Metrics).
"""

import numpy as np


# --- Confusion Matrix ---
def compute_confusion_matrix(predictions, labels, num_classes=3):
    """
    Builds a (C, C) confusion matrix. Rows represent ground truth, 
    columns represent predictions.
    """
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()

    # Cast to int64 to handle float32 rasters
    pred_flat = np.asarray(predictions).flatten().astype(np.int64)
    label_flat = np.asarray(labels).flatten().astype(np.int64)

    # Mask out-of-range labels (e.g., 254 nodata after -1 remap)
    # See Subsection 3.4.4 (UNOSAT Reference Labels)
    mask = ((label_flat >= 0) & (label_flat < num_classes) &
            (pred_flat >= 0) & (pred_flat < num_classes))

    idx = label_flat[mask] * num_classes + pred_flat[mask]
    confusion = np.bincount(idx, minlength=num_classes * num_classes)
    return confusion.reshape(num_classes, num_classes).astype(np.int64)


# --- Metrics from Confusion Matrix ---
def compute_iou_per_class(confusion, empty_union_value=0.0):
    """
    Computes per-class Intersection over Union (IoU).
    """
    # See Equation 3.17
    num_classes = confusion.shape[0]
    ious = []
    for i in range(num_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        union = tp + fp + fn
        ious.append(tp / union if union > 0 else empty_union_value)
    return ious


def compute_miou(confusion, empty_union_value=0.0):
    """
    Computes the Mean Intersection over Union (mIoU) across all classes.
    """
    # See Equation 3.18
    return float(np.mean(compute_iou_per_class(confusion, empty_union_value)))


def compute_accuracy(confusion):
    """
    Computes overall pixel accuracy.
    """
    total = confusion.sum()
    return float(np.trace(confusion) / total) if total > 0 else 0.0


def compute_precision_recall_f1(confusion):
    """
    Computes per-class precision, recall, and F1-score.
    """
    # See Equation 3.19
    num_classes = confusion.shape[0]
    precision, recall, f1 = [], [], []
    for i in range(num_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)
    return precision, recall, f1


def compute_metrics_from_confusion(confusion, empty_union_value=0.0):
    """
    Computes a full bundle of evaluation metrics from a confusion matrix.
    """
    ious = compute_iou_per_class(confusion, empty_union_value)
    precision, recall, f1 = compute_precision_recall_f1(confusion)
    
    # Macro averaging assigns equal weight to each class
    # See Equation 3.20
    return {
        'accuracy':        compute_accuracy(confusion),
        'miou':            float(np.mean(ious)),
        'ious':            ious,
        'precision':       precision,
        'recall':          recall,
        'f1':              f1,
        'macro_precision': float(np.mean(precision)),
        'macro_recall':    float(np.mean(recall)),
        'macro_f1':        float(np.mean(f1)),
        'confusion':       confusion,
    }


# --- Per-Patch IoU ---
def compute_patch_iou(pred, label, num_classes=3, empty_union_value=1.0):
    """
    Computes IoU for a single patch, averaged over classes.
    
    Note: Uses empty_union_value=1.0 (vacuous truth) because patch-level scores
    are used qualitatively in a heatmap. A patch with no PDC pixels should not 
    be penalized if there was nothing to predict. Evaluation metrics use 0.0.
    """
    # See Subsection 3.9.3 (Evaluation Scope and Metrics)
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        label_cls = (label == cls)
        intersection = (pred_cls & label_cls).sum()
        union = (pred_cls | label_cls).sum()
        ious.append(intersection / union if union > 0 else empty_union_value)
    return float(np.mean(ious))