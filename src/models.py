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
# Helper Data Structures
# =========================

@dataclass
class ConfusionCounts:
    """Container of basic confusion counts."""
    true_positive_count: int
    false_positive_count: int
    true_negative_count: int
    false_negative_count: int


@dataclass
class ResultBundle:
    """Container for test results and artifacts."""
    per_feature_metrics_data_frame: pd.DataFrame
    total_metrics_data_frame: pd.DataFrame
    row_accuracy_value: float
    entity_detection_summary: Optional[Dict[str, Any]] = None
    matched_pairs_data_frame: Optional[pd.DataFrame] = None


@dataclass
class RunContext:
    """Contextual information for a single run."""
    run_identifier: str
    started_at_timestamp: str
    configuration_hash: str