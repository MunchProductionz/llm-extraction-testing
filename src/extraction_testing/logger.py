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

from .models import RunContext, ResultBundle
from .config import RunConfig
from .utils import ensure_directory


# =========================
# Logger
# =========================

class RunLogger:
    """Responsible for writing a human-readable run log."""
    def __init__(self, log_directory_path: str):
        """Initialize with a log directory path."""
        self.log_directory_path = log_directory_path
        ensure_directory(self.log_directory_path)

    def write_log(self, run_context: RunContext, result_bundle: ResultBundle, run_config: RunConfig, note_message: Optional[str] = None) -> str:
        """Write a timestamped log file and return its path."""
        filename = f"test_run_{run_context.run_identifier}.txt"
        full_file_path_string = str(Path(self.log_directory_path) / filename)
        with open(full_file_path_string, "w", encoding="utf-8") as file_handle:
            file_handle.write(f"Run Identifier: {run_context.run_identifier}\n")
            file_handle.write(f"Started At: {run_context.started_at_timestamp}\n")
            file_handle.write(f"Configuration Hash: {run_context.configuration_hash}\n")
            file_handle.write(f"Task Type: {run_config.task_type}\n")
            file_handle.write("\n=== Total Metrics (averaged over features) ===\n")
            file_handle.write(result_bundle.total_metrics_data_frame.to_string(index=False))
            file_handle.write("\n\n=== Per-Feature Metrics (macro) ===\n")
            file_handle.write(result_bundle.per_feature_metrics_data_frame.to_string(index=False))
            file_handle.write(f"\n\nRow Accuracy: {result_bundle.row_accuracy_value:.4f}\n")
            if result_bundle.entity_detection_summary:
                file_handle.write("\n=== Entity Detection Summary ===\n")
                for key, value in result_bundle.entity_detection_summary.items():
                    file_handle.write(f"{key}: {value}\n")
            if note_message:
                file_handle.write(f"\nNote: {note_message}\n")
        return full_file_path_string
    
    
    