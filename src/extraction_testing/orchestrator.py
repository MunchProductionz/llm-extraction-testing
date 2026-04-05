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
from pydantic import BaseModel

from .config import RunConfig, TaskType
from .models import ResultBundle, RunContext
from .tests import MultiEntityExtractionTest, SingleEntityExtractionTest, SingleFeatureExtractionTest
from .utils import timestamp_string, hash_configuration, model_to_dict




# =========================
# Orchestrator / Convenience
# =========================

def build_run_context(run_config: RunConfig) -> RunContext:
    """Create a run context with identifier, timestamp, and config hash."""
    run_identifier_value = timestamp_string()
    started_at_timestamp_value = datetime.now().isoformat(timespec="seconds")
    configuration_hash_value = hash_configuration(model_to_dict(run_config))
    return RunContext(run_identifier_value, started_at_timestamp_value, configuration_hash_value)


def evaluate(predicted_records: List[BaseModel], gold_records: List[BaseModel], run_config: RunConfig) -> ResultBundle:
    """Convenience entry point to evaluate based on task type."""
    if run_config.task_type == TaskType.MULTI_ENTITY:
        tester = MultiEntityExtractionTest(run_config)
    elif run_config.task_type == TaskType.SINGLE_ENTITY:
        tester = SingleEntityExtractionTest(run_config)
    elif run_config.task_type == TaskType.SINGLE_FEATURE:
        tester = SingleFeatureExtractionTest(run_config)
    else:
        raise ValueError(f"Unsupported task type: {run_config.task_type}")
    return tester.test(predicted_records, gold_records)
