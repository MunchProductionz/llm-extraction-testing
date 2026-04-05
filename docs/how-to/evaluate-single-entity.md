# Evaluate a Single-Entity Task

## Scenario

You have one structured entity per record and each entity has several fields. A common example is extracting multiple fields from one article, form, or document.

## Runnable example

```python
from typing import Optional

from pydantic import BaseModel

from extraction_testing import FeatureRule, RunConfig, TaskType, evaluate


class ArticleRecord(BaseModel):
    row_identifier: int
    headline_text: str
    author_name: str
    publish_date: Optional[str]


predicted_records = [
    ArticleRecord(
        row_identifier=1,
        headline_text="breaking market rally",
        author_name="J. Smith",
        publish_date="2024-06-02",
    ),
    ArticleRecord(
        row_identifier=2,
        headline_text="Local Sports Win",
        author_name="Alex Coach",
        publish_date="2024-06-05",
    ),
]

gold_records = [
    ArticleRecord(
        row_identifier=1,
        headline_text="Breaking: Market Rally",
        author_name="John Smith",
        publish_date="2024-06-01",
    ),
    ArticleRecord(
        row_identifier=2,
        headline_text="Local Sports Win",
        author_name="A. Coach",
        publish_date="2024-06-01",
    ),
]

run_config = RunConfig(
    task_type=TaskType.SINGLE_ENTITY,
    feature_rules=[
        FeatureRule(feature_name="headline_text", feature_type="text"),
        FeatureRule(
            feature_name="author_name",
            feature_type="text",
            alias_map={"J. Smith": "John Smith", "Alex Coach": "A. Coach"},
        ),
        FeatureRule(
            feature_name="publish_date",
            feature_type="date",
            date_tolerance_days=1,
        ),
    ],
    index_key_name="row_identifier",
)

result_bundle = evaluate(predicted_records, gold_records, run_config)

print(result_bundle.per_feature_metrics_data_frame)
print(result_bundle.total_metrics_data_frame)
print("row_accuracy:", result_bundle.row_accuracy_value)
```

## What to expect

- the two rows are aligned by `row_identifier`
- headline and author should match on both rows after normalization and aliasing
- `publish_date` should match on row `1` because it is within one day
- `publish_date` should fail on row `2` because it is four days away

That means:

- `row_accuracy_value` should be `0.5`, because only the first row is fully correct
- `publish_date` should have weaker metrics than `headline_text` and `author_name`

## When this guide applies

Use this workflow when:

- each record maps to one known entity
- you need per-field scoring across several fields
- alignment should happen by an explicit key, not similarity matching

## Related concepts

- [Task Types](../concepts/task-types.md)
- [Feature Rules](../concepts/feature-rules.md)
- [Metrics](../concepts/metrics.md)
