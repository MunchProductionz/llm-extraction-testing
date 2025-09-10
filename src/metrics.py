from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime, date
import math
import random
import string
import hashlib

import pandas as pd
import numpy as np
from pydantic import BaseModel, validator

from src.models import ConfusionCounts
from src.utils import normalize_label_for_metrics


# =========================
# Metrics Computation
# =========================

def compute_confusion_counts_for_one_vs_rest(gold_labels: List[Any], predicted_labels: List[Any], positive_label: Any) -> ConfusionCounts:
    """Compute confusion counts for one-vs-rest of a single class."""
    true_positive_count = 0
    false_positive_count = 0
    true_negative_count = 0
    false_negative_count = 0
    for gold_value, predicted_value in zip(gold_labels, predicted_labels):
        if predicted_value == positive_label and gold_value == positive_label:
            true_positive_count += 1
        elif predicted_value == positive_label and gold_value != positive_label:
            false_positive_count += 1
        elif predicted_value != positive_label and gold_value == positive_label:
            false_negative_count += 1
        else:
            true_negative_count += 1
    return ConfusionCounts(true_positive_count, false_positive_count, true_negative_count, false_negative_count)


def safe_divide(numerator_value: float, denominator_value: float) -> float:
    """Divide safely with zero guard."""
    if denominator_value == 0:
        return 0.0
    return numerator_value / denominator_value


def precision_from_counts(confusion_counts: ConfusionCounts) -> float:
    """Compute precision."""
    return safe_divide(confusion_counts.true_positive_count, (confusion_counts.true_positive_count + confusion_counts.false_positive_count))


def recall_from_counts(confusion_counts: ConfusionCounts) -> float:
    """Compute recall."""
    return safe_divide(confusion_counts.true_positive_count, (confusion_counts.true_positive_count + confusion_counts.false_negative_count))


def specificity_from_counts(confusion_counts: ConfusionCounts) -> float:
    """Compute specificity."""
    return safe_divide(confusion_counts.true_negative_count, (confusion_counts.true_negative_count + confusion_counts.false_positive_count))


def f1_from_counts(confusion_counts: ConfusionCounts) -> float:
    """Compute F1 score."""
    precision_value = precision_from_counts(confusion_counts)
    recall_value = recall_from_counts(confusion_counts)
    if precision_value + recall_value == 0:
        return 0.0
    return 2 * precision_value * recall_value / (precision_value + recall_value)

def compute_multiclass_metrics(
    gold_labels: List[Any],
    predicted_labels: List[Any],
    positive_label: Optional[str] = None
) -> Dict[str, float]:
    """Compute macro precision/recall/F1/specificity and micro accuracy (robust to None)."""
    # Normalize labels so None becomes a comparable sentinel
    gold_normalized = [normalize_label_for_metrics(v) for v in gold_labels]
    predicted_normalized = [normalize_label_for_metrics(v) for v in predicted_labels]

    # Build a deterministic, safely sortable label list
    unique_label_list = sorted(set(gold_normalized + predicted_normalized), key=lambda x: str(x))

    if not unique_label_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "specificity": 0.0, "micro_accuracy": 0.0}

    # Per-class one-vs-rest
    per_class_metrics = []
    for label_value in unique_label_list:
        counts = compute_confusion_counts_for_one_vs_rest(gold_normalized, predicted_normalized, label_value)
        per_class_metrics.append((
            precision_from_counts(counts),
            recall_from_counts(counts),
            f1_from_counts(counts),
            specificity_from_counts(counts),
        ))

    precision_macro = float(np.mean([m[0] for m in per_class_metrics]))
    recall_macro    = float(np.mean([m[1] for m in per_class_metrics]))
    f1_macro        = float(np.mean([m[2] for m in per_class_metrics]))
    specificity_macro = float(np.mean([m[3] for m in per_class_metrics]))
    micro_accuracy  = float(np.mean([1 if g == p else 0 for g, p in zip(gold_normalized, predicted_normalized)]))

    # Optional binary report for a provided positive_label (normalize it too)
    if positive_label is not None:
        positive_label_normalized = normalize_label_for_metrics(positive_label)
        if positive_label_normalized in unique_label_list:
            counts = compute_confusion_counts_for_one_vs_rest(gold_normalized, predicted_normalized, positive_label_normalized)
            return {
                "precision":   precision_from_counts(counts),
                "recall":      recall_from_counts(counts),
                "f1":          f1_from_counts(counts),
                "specificity": specificity_from_counts(counts),
                "micro_accuracy": micro_accuracy,
            }

    return {
        "precision": precision_macro,
        "recall": recall_macro,
        "f1": f1_macro,
        "specificity": specificity_macro,
        "micro_accuracy": micro_accuracy,
    }


def compute_row_accuracy(feature_equality_data_frame: pd.DataFrame) -> float:
    """Compute fraction of rows where all feature equality flags are True."""
    if feature_equality_data_frame.shape[0] == 0:
        return 0.0
    row_all_true_series = feature_equality_data_frame.all(axis=1)
    return float(row_all_true_series.mean())




# =========================
# Result Assembly Helpers
# =========================

def make_per_feature_metrics_data_frame(per_feature_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create a tidy DataFrame for per-feature metrics."""
    record_list: List[Dict[str, Any]] = []
    for feature_name, metrics_dictionary in per_feature_metrics.items():
        record = {"feature_name": feature_name}
        record.update(metrics_dictionary)
        record_list.append(record)
    return pd.DataFrame.from_records(record_list)


def make_total_metrics_data_frame(total_metrics: Dict[str, float]) -> pd.DataFrame:
    """Create a single-row DataFrame for total metrics."""
    return pd.DataFrame([total_metrics])
