# `models`

This module defines the lightweight dataclasses returned by the evaluation pipeline.

## `ConfusionCounts`

Fields:

- `true_positive_count`
- `false_positive_count`
- `true_negative_count`
- `false_negative_count`

Used internally by the metric helpers.

## `ResultBundle`

Fields:

- `per_feature_metrics_data_frame`
- `total_metrics_data_frame`
- `row_accuracy_value`
- `entity_detection_summary=None`
- `matched_pairs_data_frame=None`

Notes:

- returned by `evaluate()`
- `entity_detection_summary` is normally populated only for `MULTI_ENTITY`
- `matched_pairs_data_frame` currently contains pair indices and similarity scores for multi-entity runs

## `RunContext`

Fields:

- `run_identifier`
- `started_at_timestamp`
- `configuration_hash`

Used mainly for logging and run traceability.

## Generated API details

::: extraction_testing.models
    options:
      members:
        - ConfusionCounts
        - ResultBundle
        - RunContext
      show_root_heading: false
