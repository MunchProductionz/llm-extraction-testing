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

from .config import FeatureRule
from .config import MISSING_LABEL_SENTINEL


# =========================
# Utility Functions
# =========================

def ensure_directory(directory_path_string: str) -> None:
    """Ensure a directory exists."""
    Path(directory_path_string).mkdir(parents=True, exist_ok=True)


def timestamp_string() -> str:
    """Return a timestamp string safe for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _flatten_config_for_hashing(object_value: Any, key_prefix: str = "") -> List[Tuple[str, Any]]:
    """Flatten nested configuration structures for hashing."""
    flattened_items: List[Tuple[str, Any]] = []
    if isinstance(object_value, dict):
        for key in sorted(object_value.keys()):
            flattened_items.extend(_flatten_config_for_hashing(object_value[key], key_prefix + str(key) + "."))
    elif isinstance(object_value, list):
        for index, inner_value in enumerate(object_value):
            flattened_items.extend(_flatten_config_for_hashing(inner_value, key_prefix + f"{index}."))
    elif isinstance(object_value, BaseModel):
        flattened_items.extend(_flatten_config_for_hashing(object_value.dict(), key_prefix))
    else:
        flattened_items.append((key_prefix.rstrip("."), object_value))
    return flattened_items


def hash_configuration(configuration_dictionary: Dict[str, Any]) -> str:
    """Create a deterministic hash for configuration."""
    serialized = repr(sorted(_flatten_config_for_hashing(configuration_dictionary))).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:12]


def record_list_to_data_frame(record_list: List[BaseModel]) -> pd.DataFrame:
    """Convert a list of Pydantic models to a pandas DataFrame."""
    rows: List[Dict[str, Any]] = []
    for record in record_list:
        if hasattr(record, "model_dump"):
            rows.append(record.model_dump())
        elif hasattr(record, "dict"):
            rows.append(record.dict())
        else:
            rows.append(dict(record))
    return pd.DataFrame(rows)

def normalize_label_for_metrics(value: Any) -> Any:
    """Return a metrics-safe label (map None to a sentinel)."""
    return MISSING_LABEL_SENTINEL if value is None else value

def normalize_text(raw_value: str, casefold_text: bool, strip_text: bool, remove_punctuation: bool) -> str:
    """Normalize text by casefolding, stripping, and punctuation removal."""
    if raw_value is None:
        return ""
    normalized_text_value = str(raw_value)
    if casefold_text:
        normalized_text_value = normalized_text_value.casefold()
    if strip_text:
        normalized_text_value = normalized_text_value.strip()
    if remove_punctuation:
        normalized_text_value = normalized_text_value.translate(str.maketrans("", "", string.punctuation))
    normalized_text_value = " ".join(normalized_text_value.split())
    return normalized_text_value


def apply_alias_map(raw_value: Any, alias_map: Optional[Dict[str, str]]) -> Any:
    """Apply alias mapping when provided."""
    if alias_map is None:
        return raw_value
    return alias_map.get(str(raw_value), raw_value)

def normalize_number(raw_value: Any, numeric_rounding_digits: Optional[int]) -> Any:
    """Normalize numeric value via rounding if configured; return None if parsing fails."""
    if raw_value is None or (isinstance(raw_value, float) and (math.isnan(raw_value) or math.isinf(raw_value))):
        return raw_value
    try:
        numeric_value = float(raw_value)
    except Exception:
        return None  # treat unparsable (e.g., "") as missing
    if numeric_rounding_digits is not None:
        return round(numeric_value, numeric_rounding_digits)
    return numeric_value

def parse_date(raw_value: Any) -> Optional[date]:
    """Parse ISO date string to date object."""
    if raw_value is None:
        return None
    try:
        if isinstance(raw_value, date):
            return raw_value
        return datetime.fromisoformat(str(raw_value)).date()
    except Exception:
        return None

def are_both_values_missing(predicted_value: Any, gold_value: Any) -> bool:
    """Return True if both values are missing (None)."""
    return predicted_value is None and gold_value is None

def is_any_value_missing(predicted_value: Any, gold_value: Any) -> bool:
    """Return True if either value is missing (None)."""
    return (predicted_value is None) or (gold_value is None)


def values_are_equal(predicted_value: Any, gold_value: Any, feature_rule: FeatureRule) -> bool:
    """Check equality under feature-specific semantics."""
    rule = feature_rule
    if rule.feature_type in {"text", "category"}:
        predicted_text = normalize_text(str(apply_alias_map(predicted_value, rule.alias_map)), rule.casefold_text, rule.strip_text, rule.remove_punctuation)
        gold_text = normalize_text(str(apply_alias_map(gold_value, rule.alias_map)), rule.casefold_text, rule.strip_text, rule.remove_punctuation)
        return predicted_text == gold_text
    if rule.feature_type == "number":
        predicted_number = normalize_number(predicted_value, rule.numeric_rounding_digits)
        gold_number = normalize_number(gold_value, rule.numeric_rounding_digits)
        if are_both_values_missing(predicted_number, gold_number):
            return True
        if is_any_value_missing(predicted_number, gold_number):
            return False
        if rule.numeric_absolute_tolerance is not None:
            if abs(predicted_number - gold_number) <= rule.numeric_absolute_tolerance:
                return True
        if rule.numeric_relative_tolerance is not None and gold_number != 0:
            if abs(predicted_number - gold_number) / abs(gold_number) <= rule.numeric_relative_tolerance:
                return True
        return predicted_number == gold_number
    if rule.feature_type == "date":
        predicted_date_value = parse_date(predicted_value)
        gold_date_value = parse_date(gold_value)
        if are_both_values_missing(predicted_date_value, gold_date_value):
            return True
        if is_any_value_missing(predicted_date_value, gold_date_value):
            return False
        if rule.date_tolerance_days is not None:
            return abs((predicted_date_value - gold_date_value).days) <= rule.date_tolerance_days
        return predicted_date_value == gold_date_value
    return predicted_value == gold_value


def compute_text_similarity(text_a: str, text_b: str) -> float:
    """Compute token-set similarity between two strings in [0,1]."""
    token_set_a = set(str(text_a).casefold().translate(str.maketrans("", "", string.punctuation)).split())
    token_set_b = set(str(text_b).casefold().translate(str.maketrans("", "", string.punctuation)).split())
    if not token_set_a and not token_set_b:
        return 1.0
    if not token_set_a or not token_set_b:
        return 0.0
    intersection_size = len(token_set_a & token_set_b)
    union_size = len(token_set_a | token_set_b)
    return intersection_size / union_size


def feature_similarity_score(predicted_value: Any, gold_value: Any, feature_rule: FeatureRule) -> float:
    """Return a similarity score for one feature for matching."""
    if feature_rule.feature_type == "text":
        predicted_text = normalize_text(str(apply_alias_map(predicted_value, feature_rule.alias_map)), feature_rule.casefold_text, feature_rule.strip_text, feature_rule.remove_punctuation)
        gold_text = normalize_text(str(apply_alias_map(gold_value, feature_rule.alias_map)), feature_rule.casefold_text, feature_rule.strip_text, feature_rule.remove_punctuation)
        return compute_text_similarity(predicted_text, gold_text)
    if feature_rule.feature_type in {"category", "number", "date"}:
        return 1.0 if values_are_equal(predicted_value, gold_value, feature_rule) else 0.0
    return 1.0 if predicted_value == gold_value else 0.0