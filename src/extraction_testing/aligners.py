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

from src.extraction_testing.config import FeatureRule, MatchingConfig, RunConfig
from src.extraction_testing.utils import values_are_equal, feature_similarity_score


# =========================
# Aligners
# =========================

class BaseAligner:
    """Base class for alignment strategies."""
    def align(self, predicted_data_frame: pd.DataFrame, gold_data_frame: pd.DataFrame, feature_rules: List[FeatureRule], run_config: RunConfig) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
        """Align predicted to gold and return matched pairs and unmatched indices."""
        raise NotImplementedError()


class IndexAligner(BaseAligner):
    """Align indexed rows by exact key match."""
    def align(self, predicted_data_frame: pd.DataFrame, gold_data_frame: pd.DataFrame, feature_rules: List[FeatureRule], run_config: RunConfig) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
        """Align using index_key_name."""
        index_key_name = run_config.index_key_name
        if not index_key_name:
            raise ValueError("index_key_name must be set for indexed tasks.")
        gold_index_map: Dict[Any, int] = {}
        for gold_index, gold_row in gold_data_frame.iterrows():
            gold_index_map[gold_row.get(index_key_name)] = gold_index

        matched_pairs: List[Tuple[int, int, float]] = []
        unmatched_predicted_indices: List[int] = []
        used_gold_indices: set = set()

        for predicted_index, predicted_row in predicted_data_frame.iterrows():
            key_value = predicted_row.get(index_key_name)
            gold_index = gold_index_map.get(key_value)
            if gold_index is not None and gold_index not in used_gold_indices:
                matched_pairs.append((predicted_index, gold_index, 1.0))
                used_gold_indices.add(gold_index)
            else:
                unmatched_predicted_indices.append(predicted_index)
        unmatched_gold_indices = [i for i in gold_data_frame.index if i not in used_gold_indices]
        return matched_pairs, unmatched_predicted_indices, unmatched_gold_indices


class EntityAligner(BaseAligner):
    """Align unindexed entities using a greedy maximum similarity matching."""
    def align(self, predicted_data_frame: pd.DataFrame, gold_data_frame: pd.DataFrame, feature_rules: List[FeatureRule], run_config: RunConfig) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
        """Compute candidate pairs and choose greedy non-overlapping matches."""
        matching_config = run_config.matching_config or MatchingConfig()
        random.seed(matching_config.random_tie_breaker_seed)

        candidate_pairs: List[Tuple[int, int, float]] = []
        for predicted_index, predicted_row in predicted_data_frame.iterrows():
            for gold_index, gold_row in gold_data_frame.iterrows():
                similarity_score_value = compute_pair_similarity(predicted_row, gold_row, feature_rules, matching_config)
                if similarity_score_value >= matching_config.minimum_similarity_threshold:
                    candidate_pairs.append((predicted_index, gold_index, similarity_score_value))

        if matching_config.maximum_candidate_pairs is not None:
            candidate_pairs = sorted(candidate_pairs, key=lambda t: t[2], reverse=True)[:matching_config.maximum_candidate_pairs]

        random.shuffle(candidate_pairs)
        candidate_pairs.sort(key=lambda t: t[2], reverse=True)

        matched_predicted_indices: set = set()
        matched_gold_indices: set = set()
        matched_pairs: List[Tuple[int, int, float]] = []

        for predicted_index, gold_index, similarity_score_value in candidate_pairs:
            if predicted_index not in matched_predicted_indices and gold_index not in matched_gold_indices:
                matched_pairs.append((predicted_index, gold_index, similarity_score_value))
                matched_predicted_indices.add(predicted_index)
                matched_gold_indices.add(gold_index)

        unmatched_predicted_indices = [i for i in predicted_data_frame.index if i not in matched_predicted_indices]
        unmatched_gold_indices = [i for i in gold_data_frame.index if i not in matched_gold_indices]
        return matched_pairs, unmatched_predicted_indices, unmatched_gold_indices
    

def compute_pair_similarity(predicted_row: pd.Series, gold_row: pd.Series, feature_rules: List[FeatureRule], matching_config: MatchingConfig) -> float:
    """Compute weighted similarity score between two rows for entity matching."""
    if matching_config.matching_mode == "exact":
        for feature_rule in feature_rules:
            if feature_rule.is_mandatory_for_matching:
                if not values_are_equal(predicted_row.get(feature_rule.feature_name), gold_row.get(feature_rule.feature_name), feature_rule):
                    return 0.0
        return 1.0

    weighted_sum_value = 0.0
    weight_total_value = 0.0
    for feature_rule in feature_rules:
        predicted_value = predicted_row.get(feature_rule.feature_name)
        gold_value = gold_row.get(feature_rule.feature_name)
        similarity_value = feature_similarity_score(predicted_value, gold_value, feature_rule)
        weighted_sum_value += feature_rule.weight_for_matching * similarity_value
        weight_total_value += feature_rule.weight_for_matching
    if weight_total_value == 0.0:
        return 0.0
    return weighted_sum_value / weight_total_value