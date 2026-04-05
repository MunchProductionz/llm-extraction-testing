# Save Logs

## Scenario

You want a human-readable record of one evaluation run that you can inspect later or attach to an experiment artifact.

## Runnable example

```python
from pydantic import BaseModel

from extraction_testing import (
    FeatureRule,
    RunConfig,
    RunLogger,
    TaskType,
    build_run_context,
    evaluate,
)


class ArticleLabel(BaseModel):
    row_identifier: int
    topic_label: str


predicted_records = [ArticleLabel(row_identifier=1, topic_label="business")]
gold_records = [ArticleLabel(row_identifier=1, topic_label="business")]

run_config = RunConfig(
    task_type=TaskType.SINGLE_FEATURE,
    feature_rules=[FeatureRule(feature_name="topic_label", feature_type="category")],
    index_key_name="row_identifier",
    log_directory_path="./logs",
)

result_bundle = evaluate(predicted_records, gold_records, run_config)
run_context = build_run_context(run_config)

logger = RunLogger(run_config.log_directory_path)
log_path = logger.write_log(
    run_context,
    result_bundle,
    run_config,
    note_message="Saved from save-logs guide",
)

print(log_path)
```

## What to expect

- the `./logs` directory is created automatically if it does not exist
- the file name follows the pattern `test_run_<run_identifier>.txt`
- the file includes total metrics, per-feature metrics, row accuracy, and metadata about the run

## When to use `build_run_context()`

Use `build_run_context(run_config)` when you want:

- a timestamp-based run identifier
- a configuration hash for reproducibility
- the metadata expected by `RunLogger.write_log()`

## Related concepts

- [Logging](../concepts/logging.md)
- [Run Config](../concepts/run-config.md)
