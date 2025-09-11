"""
Minimal demo to render and save charts from a ResultBundle-like object.

If you have real results via orchestrator.evaluate(), simply pass that
ResultBundle into the plotting functions below.
"""

import os
from typing import Any, Dict
import pandas as pd
import matplotlib.pyplot as plt

# Ensure headless-friendly backend for examples run in CI or servers
plt.switch_backend("Agg")  # no-op if already set appropriately

from extraction_testing.visualization import (  # type: ignore
    plot_total_metrics_bar,
    plot_per_feature_metrics_bar,
    plot_entity_presence_summary,
    plot_confusion_matrix_for_classification,
    save_all_charts_to_report,
)

class DummyResultBundle:
    """Tiny stand-in object with the attributes used by the plotting functions."""

    def __init__(self, total_df: pd.DataFrame, per_feature_df: pd.DataFrame, row_acc: float, summary: Dict[str, float]):
        self.total_metrics_data_frame = total_df
        self.per_feature_metrics_data_frame = per_feature_df
        self.row_accuracy_value = row_acc
        self.entity_detection_summary = summary
        self.matched_pairs_data_frame = None # optional; not used here


def build_dummy_bundle() -> DummyResultBundle:
    total = pd.DataFrame(
        [
            dict(precision=0.88, recall=0.84, f1=0.86, specificity=0.92, micro_accuracy=0.87),
        ]
    )
    per_feature = pd.DataFrame(
        [
            dict(feature_name="title", precision=0.90, recall=0.85, f1=0.875, specificity=0.93, micro_accuracy=0.88),
            dict(feature_name="author", precision=0.86, recall=0.82, f1=0.84, specificity=0.91, micro_accuracy=0.85),
            dict(feature_name="date", precision=0.92, recall=0.80, f1=0.855, specificity=0.95, micro_accuracy=0.86),
        ]
    )
    entity_summary = dict(precision=0.83, recall=0.79, f1=0.81)
    row_acc = 0.78
    return DummyResultBundle(total, per_feature, row_acc, entity_summary)


def main() -> None:
    """Run the demonstration to produce figures and saved PNGs."""
    bundle = build_dummy_bundle()


    # Generate figures (not displayed in Agg; saved below)
    fig_total = plot_total_metrics_bar(bundle)
    fig_pf_f1 = plot_per_feature_metrics_bar(bundle, metric_name="f1")
    fig_entity = plot_entity_presence_summary(bundle)


    # Save a small report to ./_viz_out (override with ET_VIZ_OUT)
    outdir = os.environ.get("ET_VIZ_OUT", "./_viz_out")
    os.makedirs(outdir, exist_ok=True)
    paths = save_all_charts_to_report(bundle, outdir)


    # Additionally, show a tiny confusion-matrix example
    gold = ["A", "B", "A", "C", "B", "C"]
    pred = ["A", "B", "C", "C", "B", "A"]
    classes = ["A", "B", "C"]
    fig_cm = plot_confusion_matrix_for_classification(gold, pred, classes)
    cm_path = os.path.join(outdir, "confusion_matrix.png")
    fig_cm.savefig(cm_path, bbox_inches="tight", dpi=150)


    # Close figures to free memory in headless runs
    plt.close(fig_total)
    plt.close(fig_pf_f1)
    plt.close(fig_entity)
    plt.close(fig_cm)


    print("Saved charts:")
    for key, path in paths.items():
        print(f" {key}: {path}")
    print(f" confusion_matrix: {cm_path}")
    

if __name__ == "__main__":
    main()