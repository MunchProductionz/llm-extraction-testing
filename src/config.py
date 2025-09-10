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


# =========================
# Enums and Config Models
# =========================

class TaskType(str, Enum):
    """Enumeration of supported task types."""
    ENTITY_EXTRACTION = "ENTITY_EXTRACTION"
    MULTI_FEATURE = "MULTI_FEATURE"
    CLASSIFICATION = "CLASSIFICATION"


class FeatureRule(BaseModel):
    """Configuration for how to compare a single feature."""
    feature_name: str
    feature_type: str  # "text", "number", "date", "category"
    is_mandatory_for_matching: bool = True
    weight_for_matching: float = 1.0

    casefold_text: bool = True
    strip_text: bool = True
    remove_punctuation: bool = True

    alias_map: Optional[Dict[str, str]] = None

    numeric_rounding_digits: Optional[int] = None
    numeric_absolute_tolerance: Optional[float] = None
    numeric_relative_tolerance: Optional[float] = None

    date_tolerance_days: Optional[int] = None

    @validator("feature_type")
    def validate_feature_type(cls, value: str) -> str:
        """Validate feature_type."""
        allowed = {"text", "number", "date", "category"}
        if value not in allowed:
            raise ValueError(f"feature_type must be one of {allowed}")
        return value


class MatchingConfig(BaseModel):
    """Configuration for entity matching."""
    matching_mode: str = "weighted"  # "exact" or "weighted"
    minimum_similarity_threshold: float = 0.5
    maximum_candidate_pairs: Optional[int] = None
    random_tie_breaker_seed: int = 13


class ClassificationConfig(BaseModel):
    """Configuration for classification reporting."""
    positive_label: Optional[str] = None
    average_strategy: str = "macro"  # "macro" or "micro" (exposed for future use)


class RunConfig(BaseModel):
    """Top-level run configuration."""
    task_type: TaskType
    feature_rules: List[FeatureRule]
    index_key_name: Optional[str] = None
    grouping_key_names: Optional[List[str]] = None
    log_directory_path: str = "./logs"
    matching_config: Optional[MatchingConfig] = None
    classification_config: Optional[ClassificationConfig] = None