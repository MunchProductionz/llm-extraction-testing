# `visualization`

This module contains optional matplotlib-based plotting helpers.

## Plotting helpers

Main functions:

- `plot_total_metrics_bar(result_bundle)`
- `plot_per_feature_metrics_bar(result_bundle, metric_name="f1")`
- `plot_entity_presence_summary(result_bundle)`
- `plot_confusion_matrix_for_classification(gold_labels, predicted_labels, class_names)`
- `plot_metric_by_group(per_feature_metrics_data_frame, group_column_name, metric_name)`
- `save_all_charts_to_report(result_bundle, output_directory_path)`

Typical return type:

- most plotting functions return `matplotlib.figure.Figure`
- `save_all_charts_to_report()` returns `dict[str, str]`

Important current caveats:

- `plot_entity_presence_summary()` expects summary keys named `precision`, `recall`, and `f1`
- the multi-entity evaluator currently emits `precision_entities`, `recall_entities`, and `f1_entities`
- `extract_labels_for_feature()` expects enriched matched-pair columns that are not produced by the current evaluator output

## Error conditions

- invalid `metric_name` for per-feature plots raises `ValueError`
- confusion-matrix plotting raises `ValueError` for length mismatches, empty class lists, or missing observed labels

## Generated API details

::: extraction_testing.visualization
    options:
      members:
        - plot_total_metrics_bar
        - plot_per_feature_metrics_bar
        - plot_entity_presence_summary
        - plot_confusion_matrix_for_classification
        - plot_metric_by_group
        - extract_labels_for_feature
        - save_all_charts_to_report
      show_root_heading: false
