# Run Config

`RunConfig` is the top-level configuration object passed into `evaluate()`. It tells the orchestrator which evaluator to run and how fields should be compared.

## `RunConfig` fields

| Field | Type | Default | Required | Notes |
|---|---|---|---|---|
| `task_type` | `TaskType` | required | yes | Chooses the evaluator |
| `feature_rules` | `list[FeatureRule]` | required | yes | The fields to compare |
| `index_key_name` | `str \| None` | `None` | indexed tasks only | Required for `SINGLE_FEATURE` and `SINGLE_ENTITY` |
| `grouping_key_names` | `list[str] \| None` | `None` | no | Present in the model, not currently used in the runtime path |
| `log_directory_path` | `str` | `"./logs"` | no | Used by `RunLogger` |
| `matching_config` | `MatchingConfig \| None` | `None` | multi-entity only | Defaults are applied if omitted |
| `classification_config` | `ClassificationConfig \| None` | `None` | no | Optional label-reporting behavior |

## `MatchingConfig`

`MatchingConfig` is used by `MULTI_ENTITY` tasks.

| Field | Type | Default | Notes |
|---|---|---|---|
| `matching_mode` | `str` | `"weighted"` | Current implementation expects `"weighted"` or `"exact"` |
| `minimum_similarity_threshold` | `float` | `0.5` | Candidate pairs below this score are discarded |
| `maximum_candidate_pairs` | `int \| None` | `None` | If set, only the top-scoring candidate pairs are kept before greedy matching |
| `random_tie_breaker_seed` | `int` | `13` | Used to make equal-score tie-breaking deterministic |

### Validation note

Unlike `FeatureRule.feature_type`, `MatchingConfig` fields are not currently validated beyond normal Pydantic type conversion. Invalid `matching_mode` values do not raise at model creation time; they fall through to the weighted-similarity path.

## `ClassificationConfig`

`ClassificationConfig` affects how feature-level label metrics are reported.

| Field | Type | Default | Notes |
|---|---|---|---|
| `positive_label` | `str \| None` | `None` | If set and observed, the metric function returns one-vs-rest metrics for that label instead of macro metrics |
| `average_strategy` | `str` | `"macro"` | Present in the model but not currently consumed by the metric computation path |

## Task-specific requirements

### `SINGLE_FEATURE`

Use:

- `task_type=TaskType.SINGLE_FEATURE`
- exactly one `FeatureRule`
- `index_key_name` set

Current runtime failures:

- missing `index_key_name` raises `ValueError`
- more than one feature rule raises `ValueError`

### `SINGLE_ENTITY`

Use:

- `task_type=TaskType.SINGLE_ENTITY`
- one or more `FeatureRule` objects
- `index_key_name` set

Current runtime failure:

- missing `index_key_name` raises `ValueError`

### `MULTI_ENTITY`

Use:

- `task_type=TaskType.MULTI_ENTITY`
- one or more `FeatureRule` objects
- optional `matching_config`

`index_key_name` is not required because rows are matched by similarity rather than by an explicit join key.

## Configuration interactions

- `task_type` decides which evaluator class the orchestrator instantiates.
- `feature_rules` affect both comparison semantics and, for multi-entity tasks, similarity scoring.
- `matching_config` only matters for `MULTI_ENTITY`.
- `classification_config.positive_label` changes per-feature precision/recall/F1/specificity output from macro to one-vs-rest for that label when the label is present.
- `log_directory_path` does nothing by itself; it is only used if you instantiate `RunLogger`.

## Common invalid or misleading configurations

- Setting `matching_config` for an indexed task.
  It is harmless but unused.

- Omitting `index_key_name` for `SINGLE_FEATURE` or `SINGLE_ENTITY`.
  The evaluator will fail at runtime.

- Passing several feature rules to `SINGLE_FEATURE`.
  The evaluator expects exactly one feature.

- Assuming `grouping_key_names` changes result aggregation.
  It is currently not used in the execution path.

- Assuming `average_strategy` switches between macro and micro reporting.
  The current metric path does not consume that field.

## Example

```python
from extraction_testing import FeatureRule, MatchingConfig, RunConfig, TaskType

run_config = RunConfig(
    task_type=TaskType.MULTI_ENTITY,
    feature_rules=[
        FeatureRule(feature_name="contract_title", feature_type="text", weight_for_matching=2.0),
        FeatureRule(feature_name="contract_amount", feature_type="number", numeric_absolute_tolerance=100.0),
    ],
    matching_config=MatchingConfig(
        matching_mode="weighted",
        minimum_similarity_threshold=0.6,
    ),
    log_directory_path="./logs",
)
```

## Related pages

- [Task Types](task-types.md)
- [Feature Rules](feature-rules.md)
- [Alignment](alignment.md)
