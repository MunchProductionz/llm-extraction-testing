from typing import Any, Dict, List, Optional
import json
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# Use a headless-friendly backend; Streamlit renders figures via st.pyplot
plt.switch_backend("Agg")

from extraction_testing.visualization import (
    plot_total_metrics_bar,
    plot_per_feature_metrics_bar,
    plot_entity_presence_summary,
    plot_confusion_matrix_for_classification,
    save_all_charts_to_report,
)

class SimpleBundle:
    """Minimal bundle wrapper for Streamlit demos."""


    def __init__(self, total_df: pd.DataFrame, per_feature_df: pd.DataFrame, row_acc: Optional[float], summary: Optional[Dict[str, float]]):
        self.total_metrics_data_frame = total_df
        self.per_feature_metrics_data_frame = per_feature_df
        self.row_accuracy_value = row_acc
        self.entity_detection_summary = summary
        self.matched_pairs_data_frame = None


def _demo_bundle() -> SimpleBundle:
    """Return a deterministic demo bundle."""
    total = pd.DataFrame([{
    "precision": 0.88, "recall": 0.84, "f1": 0.86, "specificity": 0.92, "micro_accuracy": 0.87
    }])
    per_feature = pd.DataFrame([
    {"feature_name": "title", "precision": 0.90, "recall": 0.85, "f1": 0.875, "specificity": 0.93, "micro_accuracy": 0.88},
    {"feature_name": "author", "precision": 0.86, "recall": 0.82, "f1": 0.84, "specificity": 0.91, "micro_accuracy": 0.85},
    {"feature_name": "date", "precision": 0.92, "recall": 0.80, "f1": 0.855, "specificity": 0.95, "micro_accuracy": 0.86},
    ])
    summary = {"precision": 0.83, "recall": 0.79, "f1": 0.81}
    return SimpleBundle(total, per_feature, 0.78, summary)


st.set_page_config(page_title="extraction_testing – Visualization", layout="wide")


st.title("extraction_testing – Visualization")
mode = st.sidebar.selectbox("Data source", ["Demo", "Upload CSV/JSON"], index=0)


if mode == "Demo":
    bundle = _demo_bundle()
else:
    st.sidebar.markdown("**Upload per-feature metrics CSV** (must include 'feature_name' and metric columns)")
    per_feature_file = st.sidebar.file_uploader("per_feature_metrics_data_frame.csv", type=["csv"], key="pf")


    st.sidebar.markdown("**Upload total metrics CSV** (single row with metric columns)")
    total_file = st.sidebar.file_uploader("total_metrics_data_frame.csv", type=["csv"], key="total")


    st.sidebar.markdown("**Upload entity summary JSON (optional)**")
    entity_file = st.sidebar.file_uploader("entity_detection_summary.json", type=["json"], key="ent")


    per_feature_df = pd.read_csv(per_feature_file) if per_feature_file else pd.DataFrame()
    total_df = pd.read_csv(total_file) if total_file else pd.DataFrame()
    entity_summary = json.load(entity_file) if entity_file else None

    # Row accuracy optionally as a number field
    row_accuracy_value = st.sidebar.number_input("Row accuracy (optional)", min_value=0.0, max_value=1.0, value=0.0)
    bundle = SimpleBundle(total_df, per_feature_df, row_accuracy_value if row_accuracy_value > 0 else None, entity_summary)


metric = st.sidebar.selectbox("Per-feature metric", ["f1", "precision", "recall", "specificity", "micro_accuracy"], index=0)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Total metrics")
    fig = plot_total_metrics_bar(bundle)
    st.pyplot(fig)

with col2:
    st.subheader("Per-feature metrics")
    fig = plot_per_feature_metrics_bar(bundle, metric_name=metric)
    st.pyplot(fig)

st.subheader("Entity presence summary")
fig = plot_entity_presence_summary(bundle)
st.pyplot(fig)

st.subheader("Confusion matrix (optional)")
cm_gold = st.sidebar.text_input("Gold labels (comma-separated)", value="A,B,A,C,B,C")
cm_pred = st.sidebar.text_input("Predicted labels (comma-separated)", value="A,B,C,C,B,A")
cm_classes = st.sidebar.text_input("Class names (comma-separated)", value="A,B,C")
if st.sidebar.button("Plot confusion matrix"):
    gold = [x.strip() for x in cm_gold.split(",") if x.strip()]
    pred = [x.strip() for x in cm_pred.split(",") if x.strip()]
    classes = [x.strip() for x in cm_classes.split(",") if x.strip()]
    try:
        fig = plot_confusion_matrix_for_classification(gold, pred, classes)
        st.pyplot(fig)
    except Exception as e:
        st.error(str(e))

if st.sidebar.button("Save charts to report"):
    outdir = st.sidebar.text_input("Output dir", value="./_viz_out") or "./_viz_out"
    paths = save_all_charts_to_report(bundle, outdir)
    st.success("Saved charts:")
    for k, p in paths.items():
        st.write(f"**{k}** → `{p}`")


