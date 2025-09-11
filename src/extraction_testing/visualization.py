from typing import Any, Dict, List, Tuple, Optional
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    # Prefer using package utilities when available
    from .utils import ensure_directory, timestamp_string  # type: ignore
except Exception:
    # Fallbacks if utilities are not importable in some contexts (e.g., isolated tests)
    def ensure_directory(path: str) -> None:
        """Ensure a directory exists."""
        os.makedirs(path, exist_ok=True)

    def timestamp_string() -> str:
        """Return a filesystem-friendly timestamp string."""
        # YYYYmmdd_HHMMSS
        import datetime as _dt

        return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _figure_with_note(title: str, message: str) -> plt.Figure:
    """Create a figure with a centered textual note."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12,
        wrap=True,
    )
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def _validate_metric_name(metric_name: str, allowed: List[str]) -> None:
    """Raise if metric_name not allowed."""
    if metric_name not in allowed:
        raise ValueError(f"metric_name must be one of {allowed}, got '{metric_name}'.")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce a Series to numeric, preserving NaN for non-convertible values."""
    return pd.to_numeric(series, errors="coerce")


def _sorted_by_metric(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Return a copy sorted by metric descending, then by feature_name ascending for determinism."""
    df2 = df.copy()
    if "feature_name" in df2.columns:
        df2 = df2.sort_values(
            by=[metric_name, "feature_name"],
            ascending=[False, True],
            kind="mergesort",  # stable sort
        )
    else:
        df2 = df2.sort_values(by=[metric_name], ascending=[False], kind="mergesort")
    return df2


def plot_total_metrics_bar(result_bundle: Any) -> plt.Figure:
    """Plot total metrics (precision, recall, f1, specificity, micro_accuracy, row_accuracy)."""
    title = "Total Metrics"
    total_df: Optional[pd.DataFrame] = getattr(result_bundle, "total_metrics_data_frame", None)
    row_accuracy_value: Optional[float] = getattr(result_bundle, "row_accuracy_value", None)

    metrics_to_show: List[str] = ["precision", "recall", "f1", "specificity", "micro_accuracy"]
    values: Dict[str, float] = {}

    if isinstance(total_df, pd.DataFrame) and not total_df.empty:
        row0 = total_df.iloc[0]
        for m in metrics_to_show:
            if m in total_df.columns:
                v = row0[m]
                try:
                    values[m] = float(v)
                except Exception:
                    values[m] = float("nan")

    # Add row accuracy if present
    if row_accuracy_value is not None:
        try:
            values["row_accuracy"] = float(row_accuracy_value)
        except Exception:
            values["row_accuracy"] = float("nan")

    if not values:
        return _figure_with_note(title, "No total metrics available to display.")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    names = list(values.keys())
    data = [values[k] for k in names]
    ax.bar(range(len(names)), data)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_per_feature_metrics_bar(result_bundle: Any, metric_name: str = "f1") -> plt.Figure:
    """Plot a per-feature bar chart for the selected metric."""
    title = f"Per-Feature {metric_name.upper()}"

    pf_df: Optional[pd.DataFrame] = getattr(result_bundle, "per_feature_metrics_data_frame", None)
    if not isinstance(pf_df, pd.DataFrame) or pf_df.empty:
        return _figure_with_note(title, "No per-feature metrics available to display.")

    allowed = ["precision", "recall", "f1", "specificity", "micro_accuracy"]
    _validate_metric_name(metric_name, allowed)

    if "feature_name" not in pf_df.columns:
        return _figure_with_note(title, "Missing 'feature_name' column in metrics DataFrame.")

    df = pf_df.copy()
    df[metric_name] = _coerce_numeric(df[metric_name])
    df = _sorted_by_metric(df, metric_name)

    if df.empty:
        return _figure_with_note(title, "No per-feature metrics available to display.")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(len(df))
    ax.bar(x, df[metric_name].fillna(0.0).values)
    ax.set_xticks(x)
    ax.set_xticklabels(df["feature_name"].astype(str).tolist(), rotation=45, ha="right")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_entity_presence_summary(result_bundle: Any) -> plt.Figure:
    """Plot entity presence summary (precision/recall/f1) if available."""
    title = "Entity Presence Metrics"
    summary: Optional[Dict[str, Any]] = getattr(result_bundle, "entity_detection_summary", None)
    if not summary or not isinstance(summary, dict):
        return _figure_with_note(title, "No entity presence summary available to display.")

    keys = ["precision", "recall", "f1"]
    data = []
    for k in keys:
        v = summary.get(k, None)
        try:
            data.append(float(v))
        except Exception:
            data.append(float("nan"))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(range(len(keys)), data)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.capitalize() for k in keys])
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrix_for_classification(
    gold_labels: List[Any],
    predicted_labels: List[Any],
    class_names: List[Any],
) -> plt.Figure:
    """Plot a confusion matrix from gold and predicted labels."""
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("gold_labels and predicted_labels must have equal length.")

    if not class_names:
        raise ValueError("class_names cannot be empty.")

    # Validate all labels exist in class_names
    observed = set(gold_labels) | set(predicted_labels)
    missing = observed - set(class_names)
    if missing:
        raise ValueError(f"class_names missing observed labels: {sorted(missing)}")

    index_map = {c: i for i, c in enumerate(class_names)}
    n = len(class_names)
    mat = np.zeros((n, n), dtype=int)
    for g, p in zip(gold_labels, predicted_labels):
        gi = index_map[g]
        pi = index_map[p]
        mat[gi, pi] += 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(c) for c in class_names], rotation=45, ha="right")
    ax.set_yticklabels([str(c) for c in class_names])

    # Annotate counts
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_metric_by_group(
    per_feature_metrics_data_frame: pd.DataFrame,
    group_column_name: str,
    metric_name: str,
) -> plt.Figure:
    """Plot mean metric by group from a pre-joined per-feature DataFrame."""
    title = f"Mean {metric_name.capitalize()} by {group_column_name}"
    df = per_feature_metrics_data_frame
    if not isinstance(df, pd.DataFrame) or df.empty:
        return _figure_with_note(title, "No data available to display.")

    if group_column_name not in df.columns:
        return _figure_with_note(title, f"Missing '{group_column_name}' column.")

    if metric_name not in df.columns:
        return _figure_with_note(title, f"Missing metric column '{metric_name}'.")

    g = (
        df[[group_column_name, metric_name]]
        .assign(**{metric_name: _coerce_numeric(df[metric_name])})
        .groupby(group_column_name, dropna=False)[metric_name]
        .mean()
        .reset_index()
    )
    if g.empty:
        return _figure_with_note(title, "No groups to display.")

    # Deterministic order
    g = g.sort_values(by=[metric_name, group_column_name], ascending=[False, True], kind="mergesort")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(len(g))
    ax.bar(x, g[metric_name].fillna(0.0).values)
    ax.set_xticks(x)
    ax.set_xticklabels(g[group_column_name].astype(str).tolist(), rotation=45, ha="right")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def extract_labels_for_feature(result_bundle: Any, feature_name: str) -> Tuple[List[Any], List[Any]]:
    """Extract gold and predicted labels for a feature from a matched pairs DataFrame."""
    df: Optional[pd.DataFrame] = getattr(result_bundle, "matched_pairs_data_frame", None)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("matched_pairs_data_frame is missing or empty in the result bundle.")

    gold_col = f"{feature_name}_gold"
    pred_col = f"{feature_name}_pred"
    missing_cols = [c for c in [gold_col, pred_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found for feature '{feature_name}': {missing_cols}")

    gold = df[gold_col].tolist()
    pred = df[pred_col].tolist()
    return gold, pred


def _save_figure(fig: plt.Figure, path: str) -> None:
    """Save a figure to a file path and close it."""
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_all_charts_to_report(result_bundle: Any, output_directory_path: str) -> Dict[str, str]:
    """Save standard charts to a timestamped folder and return their file paths."""
    ensure_directory(output_directory_path)
    folder = os.path.join(output_directory_path, f"report_{timestamp_string()}")
    ensure_directory(folder)

    outputs: Dict[str, str] = {}

    # Total metrics
    fig = plot_total_metrics_bar(result_bundle)
    p = os.path.join(folder, "total_metrics.png")
    _save_figure(fig, p)
    outputs["total_metrics"] = p

    # Per-feature F1
    fig = plot_per_feature_metrics_bar(result_bundle, metric_name="f1")
    p = os.path.join(folder, "per_feature_f1.png")
    _save_figure(fig, p)
    outputs["per_feature_f1"] = p

    # Entity presence (if any)
    try:
        fig = plot_entity_presence_summary(result_bundle)
        p = os.path.join(folder, "entity_presence.png")
        _save_figure(fig, p)
        outputs["entity_presence"] = p
    except Exception:
        # Be resilient; if something goes wrong, just skip silently.
        pass

    return outputs