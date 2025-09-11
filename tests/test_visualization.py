import os
from typing import Dict
import pandas as pd
import matplotlib


# Use a headless backend for CI
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from extraction_testing.visualization import (
    plot_total_metrics_bar,
    plot_per_feature_metrics_bar,
    plot_entity_presence_summary,
    plot_confusion_matrix_for_classification,
    save_all_charts_to_report,
)


class _Bundle:
    """Tiny helper bundle for tests."""

    def __init__(self, total_df: pd.DataFrame, per_feature_df: pd.DataFrame, row_acc: float, summary: Dict[str, float]):
        self.total_metrics_data_frame = total_df
        self.per_feature_metrics_data_frame = per_feature_df
        self.row_accuracy_value = row_acc
        self.entity_detection_summary = summary
        self.matched_pairs_data_frame = None


def _bundle() -> _Bundle:
    total = pd.DataFrame([
    dict(precision=0.5, recall=0.6, f1=0.55, specificity=0.9, micro_accuracy=0.7)
    ])
    per_feature = pd.DataFrame([
    dict(feature_name="a", precision=0.5, recall=0.6, f1=0.55, specificity=0.9, micro_accuracy=0.7),
    dict(feature_name="b", precision=0.7, recall=0.5, f1=0.58, specificity=0.8, micro_accuracy=0.65),
    ])
    summary = dict(precision=0.6, recall=0.55, f1=0.575)
    return _Bundle(total, per_feature, 0.66, summary)


def test_plot_functions_return_figures():
    b = _bundle()
    assert hasattr(plot_total_metrics_bar(b), "savefig")
    assert hasattr(plot_per_feature_metrics_bar(b, "f1"), "savefig")
    assert hasattr(plot_entity_presence_summary(b), "savefig")

def test_confusion_matrix_fig():
    fig = plot_confusion_matrix_for_classification(["A", "B", "A"], ["A", "B", "B"], ["A", "B"])
    assert hasattr(fig, "savefig")

def test_save_all_charts_to_report(tmp_path):
    b = _bundle()
    outdir = tmp_path / "out"
    mapping = save_all_charts_to_report(b, str(outdir))
    assert "total_metrics" in mapping
    assert os.path.exists(mapping["total_metrics"]) # file created
    assert os.path.getsize(mapping["total_metrics"]) > 0