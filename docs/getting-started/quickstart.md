# Quickstart

This example shows the smallest realistic `SINGLE_FEATURE` evaluation. It is the easiest entry point because it uses one indexed record list, one feature, and no entity matching.

## Minimal example

```python
from pydantic import BaseModel

from extraction_testing import (
    FeatureRule,
    RunConfig,
    TaskType,
    evaluate,
)


class ArticleLabel(BaseModel):
    row_identifier: int
    topic_label: str


predicted_records = [
    ArticleLabel(row_identifier=1, topic_label="technology"),
    ArticleLabel(row_identifier=2, topic_label="business"),
]

gold_records = [
    ArticleLabel(row_identifier=1, topic_label="tech"),
    ArticleLabel(row_identifier=2, topic_label="business"),
]

run_config = RunConfig(
    task_type=TaskType.SINGLE_FEATURE,
    feature_rules=[
        FeatureRule(
            feature_name="topic_label",
            feature_type="category",
            alias_map={"technology": "tech"},
        )
    ],
    index_key_name="row_identifier",
)

result_bundle = evaluate(predicted_records, gold_records, run_config)

print(result_bundle.per_feature_metrics_data_frame)
print(result_bundle.total_metrics_data_frame)
print(result_bundle.row_accuracy_value)
```

## What this example shows

- Records are passed in as lists of Pydantic models.
- `task_type=TaskType.SINGLE_FEATURE` means exactly one feature should be evaluated.
- `index_key_name` is required because the evaluator aligns rows by exact key match.
- `alias_map` lets equivalent category values compare as equal before metrics are computed.

## When to use a different setup

- If each indexed record has several fields to compare, use `SINGLE_ENTITY`.
- If you are comparing lists of entities with no shared row identifier, use `MULTI_ENTITY`.

See [Choosing a Task Type](task-selection.md) for a side-by-side comparison.

## Optional: write a log file

If you want a text summary on disk, add a logger step:

```python
from extraction_testing import RunLogger, build_run_context

logger = RunLogger("./logs")
log_path = logger.write_log(
    build_run_context(run_config),
    result_bundle,
    run_config,
    note_message="Quickstart run",
)
print(log_path)
```

## What to read next

- [Choosing a Task Type](task-selection.md)
- [Feature Rules](../concepts/feature-rules.md)
- [Run Config](../concepts/run-config.md)
