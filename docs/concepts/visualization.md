# Visualization

The package includes optional matplotlib-based helpers for turning a `ResultBundle` into simple charts.

## Dependency model

Visualization is optional. Install it through the visualization-capable extras:

- `uv sync --extra examples`
- or `pip install -e ".[viz]"`

## Available plotting functions

| Function | Purpose |
|---|---|
| `plot_total_metrics_bar(result_bundle)` | Plot total precision, recall, F1, specificity, micro accuracy, and row accuracy |
| `plot_per_feature_metrics_bar(result_bundle, metric_name="f1")` | Plot one metric per feature |
| `plot_entity_presence_summary(result_bundle)` | Plot entity presence summary if a compatible summary dict is present |
| `plot_confusion_matrix_for_classification(gold_labels, predicted_labels, class_names)` | Plot a confusion matrix from explicit labels |
| `plot_metric_by_group(per_feature_metrics_data_frame, group_column_name, metric_name)` | Plot mean metric by a grouping column in a pre-joined DataFrame |
| `save_all_charts_to_report(result_bundle, output_directory_path)` | Save standard charts into a timestamped report folder |

## General behavior

The plotting helpers are intentionally simple:

- one chart per figure
- deterministic sorting for per-feature and grouped plots
- y-axis constrained to `[0.0, 1.0]` for score-based charts
- empty or invalid inputs usually produce a figure with a clear note instead of failing

## Report generation

`save_all_charts_to_report(...)`:

1. ensures the output directory exists
2. creates a timestamped folder named `report_<timestamp>`
3. saves standard charts as PNG files
4. returns a mapping from chart name to file path

Current standard outputs:

- `total_metrics.png`
- `per_feature_f1.png`
- `entity_presence.png` when available

## Current implementation notes

### Entity presence summary keys

`plot_entity_presence_summary()` currently looks for summary keys named:

- `precision`
- `recall`
- `f1`

The multi-entity evaluator currently stores:

- `precision_entities`
- `recall_entities`
- `f1_entities`

So if you pass the raw `ResultBundle` from `MULTI_ENTITY` evaluation directly, this chart may not display the intended values unless you adapt the summary keys first.

### Matched-pairs helper expectations

`extract_labels_for_feature()` expects a `matched_pairs_data_frame` containing columns named like:

- `<feature_name>_gold`
- `<feature_name>_pred`

The current evaluator's `matched_pairs_data_frame` contains only indices and similarity scores, so that helper is mainly useful with custom or enriched result bundles.

## Example

```python
from extraction_testing.visualization import (
    plot_total_metrics_bar,
    plot_per_feature_metrics_bar,
    save_all_charts_to_report,
)

fig_total = plot_total_metrics_bar(result_bundle)
fig_f1 = plot_per_feature_metrics_bar(result_bundle, metric_name="f1")
paths = save_all_charts_to_report(result_bundle, "./_viz_out")
```

## Related pages

- [Metrics](metrics.md)
- [How to Generate Visualizations](../how-to/generate-visualizations.md)
