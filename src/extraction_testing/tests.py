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

from .config import RunConfig, ClassificationConfig, FeatureRule
from .models import ResultBundle
from .aligners import EntityAligner, IndexAligner
from .metrics import (
    compute_multiclass_metrics,
    make_per_feature_metrics_data_frame,
    make_total_metrics_data_frame,
    compute_row_accuracy
)
from .utils import (
    normalize_text,
    normalize_number,
    parse_date,
    apply_alias_map,
    record_list_to_data_frame,
    values_are_equal
)


# =========================
# Test Classes
# =========================

class BaseEvaluator:
    """Base test class with common helpers."""
    def __init__(self, run_config: RunConfig):
        """Store configuration for the run."""
        self.run_config = run_config
        self.feature_rule_map: Dict[str, FeatureRule] = {rule.feature_name: rule for rule in run_config.feature_rules}

    def normalize_series_for_labels(self, value_series: pd.Series, feature_rule: FeatureRule) -> List[Any]:
        """Normalize a pandas Series of values into label-like values for metrics."""
        normalized_values: List[Any] = []
        for raw_value in value_series.tolist():
            if feature_rule.feature_type in {"text", "category"}:
                normalized_text_value = normalize_text(
                    str(apply_alias_map(raw_value, feature_rule.alias_map)),
                    feature_rule.casefold_text,
                    feature_rule.strip_text,
                    feature_rule.remove_punctuation,
                )
                normalized_values.append(normalized_text_value)
            elif feature_rule.feature_type == "number":
                normalized_number_value = normalize_number(raw_value, feature_rule.numeric_rounding_digits)
                normalized_values.append(normalized_number_value)
            elif feature_rule.feature_type == "date":
                parsed_date_value = parse_date(raw_value)
                normalized_values.append(parsed_date_value.isoformat() if parsed_date_value is not None else None)
            else:
                normalized_values.append(raw_value)
        return normalized_values

    def compute_feature_equality_matrix(self, predicted_aligned_data_frame: pd.DataFrame, gold_aligned_data_frame: pd.DataFrame) -> pd.DataFrame:
        """Compute a DataFrame of per-feature equality flags for aligned rows."""
        equality_columns_dictionary: Dict[str, List[bool]] = {}
        for feature_rule in self.run_config.feature_rules:
            comparison_results: List[bool] = []
            for (_, predicted_row), (_, gold_row) in zip(predicted_aligned_data_frame.iterrows(), gold_aligned_data_frame.iterrows()):
                comparison_results.append(
                    values_are_equal(predicted_row.get(feature_rule.feature_name), gold_row.get(feature_rule.feature_name), feature_rule)
                )
            equality_columns_dictionary[feature_rule.feature_name] = comparison_results
        return pd.DataFrame(equality_columns_dictionary)

    def compute_per_feature_metrics(self, predicted_aligned_data_frame: pd.DataFrame, gold_aligned_data_frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute macro metrics per feature."""
        per_feature_metrics_dictionary: Dict[str, Dict[str, float]] = {}
        for feature_rule in self.run_config.feature_rules:
            gold_labels = self.normalize_series_for_labels(gold_aligned_data_frame[feature_rule.feature_name], feature_rule)
            predicted_labels = self.normalize_series_for_labels(predicted_aligned_data_frame[feature_rule.feature_name], feature_rule)
            classification_config = self.run_config.classification_config or ClassificationConfig()
            metrics_dictionary = compute_multiclass_metrics(gold_labels, predicted_labels, positive_label=classification_config.positive_label)
            per_feature_metrics_dictionary[feature_rule.feature_name] = metrics_dictionary
        return per_feature_metrics_dictionary

    def compute_total_metrics_from_per_feature(self, per_feature_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute total metrics as average over features of macro metrics."""
        if not per_feature_metrics:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "specificity": 0.0}
        precision_values = [m["precision"] for m in per_feature_metrics.values()]
        recall_values = [m["recall"] for m in per_feature_metrics.values()]
        f1_values = [m["f1"] for m in per_feature_metrics.values()]
        specificity_values = [m["specificity"] for m in per_feature_metrics.values()]
        return {
            "precision": float(np.mean(precision_values)),
            "recall": float(np.mean(recall_values)),
            "f1": float(np.mean(f1_values)),
            "specificity": float(np.mean(specificity_values)),
        }


class EntityExtractionTest(BaseEvaluator):
    """Entity extraction test with matching step before scoring."""
    def test(self, predicted_records: List[BaseModel], gold_records: List[BaseModel]) -> ResultBundle:
        """Run the entity extraction evaluation and return results."""
        predicted_data_frame = record_list_to_data_frame(predicted_records)
        gold_data_frame = record_list_to_data_frame(gold_records)

        aligner = EntityAligner()
        matched_pairs, unmatched_predicted_indices, unmatched_gold_indices = aligner.align(
            predicted_data_frame, gold_data_frame, self.run_config.feature_rules, self.run_config
        )

        predicted_aligned_rows: List[pd.Series] = []
        gold_aligned_rows: List[pd.Series] = []
        for predicted_index, gold_index, similarity_score_value in matched_pairs:
            predicted_aligned_rows.append(predicted_data_frame.loc[predicted_index])
            gold_aligned_rows.append(gold_data_frame.loc[gold_index])

        if predicted_aligned_rows:
            predicted_aligned_data_frame = pd.DataFrame(predicted_aligned_rows).reset_index(drop=True)
            gold_aligned_data_frame = pd.DataFrame(gold_aligned_rows).reset_index(drop=True)
        else:
            predicted_aligned_data_frame = pd.DataFrame(columns=gold_data_frame.columns)
            gold_aligned_data_frame = pd.DataFrame(columns=gold_data_frame.columns)

        per_feature_metrics_dictionary = self.compute_per_feature_metrics(predicted_aligned_data_frame, gold_aligned_data_frame)
        total_metrics_dictionary = self.compute_total_metrics_from_per_feature(per_feature_metrics_dictionary)

        feature_equality_data_frame = self.compute_feature_equality_matrix(predicted_aligned_data_frame, gold_aligned_data_frame)
        row_accuracy_value = compute_row_accuracy(feature_equality_data_frame)

        predicted_count = len(predicted_data_frame)
        gold_count = len(gold_data_frame)
        matched_count = len(matched_pairs)
        extra_predictions_count = len(unmatched_predicted_indices)
        missed_gold_count = len(unmatched_gold_indices)

        precision_entities = (matched_count / predicted_count) if predicted_count else 0.0
        recall_entities = (matched_count / gold_count) if gold_count else 0.0
        f1_entities = 0.0
        if precision_entities + recall_entities > 0:
            f1_entities = 2 * precision_entities * recall_entities / (precision_entities + recall_entities)

        entity_detection_summary_dictionary = {
            "predicted_count": predicted_count,
            "gold_count": gold_count,
            "matched_count": matched_count,
            "extra_predictions_count": extra_predictions_count,
            "missed_gold_count": missed_gold_count,
            "precision_entities": precision_entities,
            "recall_entities": recall_entities,
            "f1_entities": f1_entities,
        }

        per_feature_metrics_data_frame = make_per_feature_metrics_data_frame(per_feature_metrics_dictionary)
        total_metrics_data_frame = make_total_metrics_data_frame(total_metrics_dictionary)
        matched_pairs_data_frame = pd.DataFrame(
            [{"predicted_index": p, "gold_index": g, "similarity_score": s} for p, g, s in matched_pairs]
        )

        return ResultBundle(
            per_feature_metrics_data_frame=per_feature_metrics_data_frame,
            total_metrics_data_frame=total_metrics_data_frame,
            row_accuracy_value=row_accuracy_value,
            entity_detection_summary=entity_detection_summary_dictionary,
            matched_pairs_data_frame=matched_pairs_data_frame,
        )


class MultiFeatureExtractionTest(BaseEvaluator):
    """Multi-feature extraction test with indexed alignment."""
    def test(self, predicted_records: List[BaseModel], gold_records: List[BaseModel]) -> ResultBundle:
        """Run the multi-feature evaluation and return results."""
        if not self.run_config.index_key_name:
            raise ValueError("index_key_name must be specified for multi-feature extraction.")
        predicted_data_frame = record_list_to_data_frame(predicted_records)
        gold_data_frame = record_list_to_data_frame(gold_records)

        aligner = IndexAligner()
        matched_pairs, unmatched_predicted_indices, unmatched_gold_indices = aligner.align(
            predicted_data_frame, gold_data_frame, self.run_config.feature_rules, self.run_config
        )

        predicted_aligned_rows: List[pd.Series] = []
        gold_aligned_rows: List[pd.Series] = []
        for predicted_index, gold_index, similarity_score_value in matched_pairs:
            predicted_aligned_rows.append(predicted_data_frame.loc[predicted_index])
            gold_aligned_rows.append(gold_data_frame.loc[gold_index])

        if predicted_aligned_rows:
            predicted_aligned_data_frame = pd.DataFrame(predicted_aligned_rows).reset_index(drop=True)
            gold_aligned_data_frame = pd.DataFrame(gold_aligned_rows).reset_index(drop=True)
        else:
            predicted_aligned_data_frame = pd.DataFrame(columns=gold_data_frame.columns)
            gold_aligned_data_frame = pd.DataFrame(columns=gold_data_frame.columns)

        per_feature_metrics_dictionary = self.compute_per_feature_metrics(predicted_aligned_data_frame, gold_aligned_data_frame)
        total_metrics_dictionary = self.compute_total_metrics_from_per_feature(per_feature_metrics_dictionary)
        feature_equality_data_frame = self.compute_feature_equality_matrix(predicted_aligned_data_frame, gold_aligned_data_frame)
        row_accuracy_value = compute_row_accuracy(feature_equality_data_frame)

        per_feature_metrics_data_frame = make_per_feature_metrics_data_frame(per_feature_metrics_dictionary)
        total_metrics_data_frame = make_total_metrics_data_frame(total_metrics_dictionary)

        return ResultBundle(
            per_feature_metrics_data_frame=per_feature_metrics_data_frame,
            total_metrics_data_frame=total_metrics_data_frame,
            row_accuracy_value=row_accuracy_value,
        )


class ClassificationTest(BaseEvaluator):
    """Classification test as a single-feature indexed case."""
    def test(self, predicted_records: List[BaseModel], gold_records: List[BaseModel]) -> ResultBundle:
        """Run the classification evaluation and return results."""
        if not self.run_config.index_key_name:
            raise ValueError("index_key_name must be specified for classification.")
        predicted_data_frame = record_list_to_data_frame(predicted_records)
        gold_data_frame = record_list_to_data_frame(gold_records)

        aligner = IndexAligner()
        matched_pairs, _, _ = aligner.align(predicted_data_frame, gold_data_frame, self.run_config.feature_rules, self.run_config)

        predicted_aligned_rows: List[pd.Series] = []
        gold_aligned_rows: List[pd.Series] = []
        for predicted_index, gold_index, similarity_score_value in matched_pairs:
            predicted_aligned_rows.append(predicted_data_frame.loc[predicted_index])
            gold_aligned_rows.append(gold_data_frame.loc[gold_index])

        predicted_aligned_data_frame = pd.DataFrame(predicted_aligned_rows).reset_index(drop=True)
        gold_aligned_data_frame = pd.DataFrame(gold_aligned_rows).reset_index(drop=True)

        if len(self.run_config.feature_rules) != 1:
            raise ValueError("ClassificationTest expects exactly one feature rule.")

        feature_rule = self.run_config.feature_rules[0]
        gold_labels = self.normalize_series_for_labels(gold_aligned_data_frame[feature_rule.feature_name], feature_rule)
        predicted_labels = self.normalize_series_for_labels(predicted_aligned_data_frame[feature_rule.feature_name], feature_rule)
        classification_config = self.run_config.classification_config or ClassificationConfig()
        metrics_dictionary = compute_multiclass_metrics(gold_labels, predicted_labels, positive_label=classification_config.positive_label)

        per_feature_metrics_dictionary = {feature_rule.feature_name: metrics_dictionary}
        total_metrics_dictionary = metrics_dictionary
        row_accuracy_value = float(np.mean([1 if g == p else 0 for g, p in zip(gold_labels, predicted_labels)]))

        per_feature_metrics_data_frame = make_per_feature_metrics_data_frame(per_feature_metrics_dictionary)
        total_metrics_data_frame = make_total_metrics_data_frame(total_metrics_dictionary)

        return ResultBundle(
            per_feature_metrics_data_frame=per_feature_metrics_data_frame,
            total_metrics_data_frame=total_metrics_data_frame,
            row_accuracy_value=row_accuracy_value,
        )