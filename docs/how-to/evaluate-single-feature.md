# Evaluate a Single-Feature Task

## Scenario

You have one label-like field per record and a stable row identifier. A common example is topic classification, where each document has one predicted topic label and one gold topic label.

## Runnable example

```python
from pydantic import BaseModel

from extraction_testing import FeatureRule, RunConfig, TaskType, evaluate


class ArticleLabel(BaseModel):
    row_identifier: int
    topic_label: str


predicted_records = [
    ArticleLabel(row_identifier=1, topic_label="technology"),
    ArticleLabel(row_identifier=2, topic_label="business"),
    ArticleLabel(row_identifier=3, topic_label="sports"),
]

gold_records = [
    ArticleLabel(row_identifier=1, topic_label="tech"),
    ArticleLabel(row_identifier=2, topic_label="business"),
    ArticleLabel(row_identifier=3, topic_label="politics"),
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
print("row_accuracy:", result_bundle.row_accuracy_value)
```

## What to expect

- `per_feature_metrics_data_frame` contains one row for `topic_label`
- `total_metrics_data_frame` is the same one-feature summary in DataFrame form
- `row_accuracy_value` is the fraction of aligned rows where the final label matches exactly

In this example:

- row `1` should count as correct because the alias map converts `"technology"` to `"tech"`
- row `2` should count as correct directly
- row `3` should count as incorrect

So the row accuracy should be about `0.6667`.

## When this guide applies

Use this workflow when:

- you have exactly one feature to score
- record identity is known through a key like `row_identifier`
- you want simple per-label metrics without entity matching

## Related concepts

- [Task Types](../concepts/task-types.md)
- [Feature Rules](../concepts/feature-rules.md)
- [Run Config](../concepts/run-config.md)
