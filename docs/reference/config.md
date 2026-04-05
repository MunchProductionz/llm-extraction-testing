# `config`

This module defines the task enum and the configuration models used to control evaluation.

## Key symbols

### `TaskType`

Allowed enum values:

- `SINGLE_FEATURE`
- `SINGLE_ENTITY`
- `MULTI_ENTITY`

### `FeatureRule`

Constructor highlights:

- required: `feature_name`, `feature_type`
- validated allowed `feature_type` values: `text`, `number`, `date`, `category`
- side effect: none
- error conditions: invalid `feature_type` raises `ValueError`

### `MatchingConfig`

Constructor highlights:

- defaults to weighted matching with threshold `0.5`
- used only by `MULTI_ENTITY`
- side effect: none
- current caveat: `matching_mode` is not explicitly validated beyond normal type coercion

### `ClassificationConfig`

Constructor highlights:

- `positive_label` switches the metric function into one-vs-rest mode for that label when present
- `average_strategy` exists in the model but is not currently consumed in the metric path

### `RunConfig`

Constructor highlights:

- required: `task_type`, `feature_rules`
- `index_key_name` is required at runtime for indexed task types
- `matching_config` is relevant only for `MULTI_ENTITY`
- `log_directory_path` controls where `RunLogger` writes files
- `grouping_key_names` exists but is not currently used in the runtime path

## Generated API details

::: extraction_testing.config
    options:
      members:
        - TaskType
        - FeatureRule
        - MatchingConfig
        - ClassificationConfig
        - RunConfig
      show_root_heading: false
