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


def compute_multiclass_metrics(gold_labels: List[Any], predicted_labels: List[Any], positive_label: Optional[str] = None) -> Dict[str, float]:
    """Compute macro precision/recall/F1/specificity and micro accuracy."""
    unique_label_set = sorted(set(list(gold_labels) + list(predicted_labels)))
    if not unique_label_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "specificity": 0.0, "micro_accuracy": 0.0}

    per_class_metrics_list: List[Tuple[float, float, float, float]] = []
    for label_value in unique_label_set:
        counts = compute_confusion_counts_for_one_vs_rest(gold_labels, predicted_labels, label_value)
        per_class_metrics_list.append((
            precision_from_counts(counts),
            recall_from_counts(counts),
            f1_from_counts(counts),
            specificity_from_counts(counts),
        ))

    precision_macro = float(np.mean([m[0] for m in per_class_metrics_list]))
    recall_macro = float(np.mean([m[1] for m in per_class_metrics_list]))
    f1_macro = float(np.mean([m[2] for m in per_class_metrics_list]))
    specificity_macro = float(np.mean([m[3] for m in per_class_metrics_list]))
    micro_accuracy = float(np.mean([1 if g == p else 0 for g, p in zip(gold_labels, predicted_labels)]))

    if positive_label is not None and positive_label in unique_label_set:
        counts_positive = compute_confusion_counts_for_one_vs_rest(gold_labels, predicted_labels, positive_label)
        return {
            "precision": precision_from_counts(counts_positive),
            "recall": recall_from_counts(counts_positive),
            "f1": f1_from_counts(counts_positive),
            "specificity": specificity_from_counts(counts_positive),
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
