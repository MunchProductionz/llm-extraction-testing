# Configure Feature Rules

## Scenario

You want one evaluation run to compare several feature types with the right normalization rules for each field.

## Runnable example

```python
from typing import Optional

from pydantic import BaseModel

from extraction_testing import FeatureRule, RunConfig, TaskType, evaluate


class ArticleRecord(BaseModel):
    row_identifier: int
    headline_text: str
    view_count: Optional[int]
    publish_date: Optional[str]
    topic_label: str


feature_rules = [
    FeatureRule(
        feature_name="headline_text",
        feature_type="text",
        casefold_text=True,
        strip_text=True,
        remove_punctuation=True,
    ),
    FeatureRule(
        feature_name="view_count",
        feature_type="number",
        numeric_absolute_tolerance=5,
        numeric_rounding_digits=0,
    ),
    FeatureRule(
        feature_name="publish_date",
        feature_type="date",
        date_tolerance_days=1,
    ),
    FeatureRule(
        feature_name="topic_label",
        feature_type="category",
        alias_map={"technology": "tech"},
    ),
]

predicted_records = [
    ArticleRecord(
        row_identifier=1,
        headline_text="breaking market rally",
        view_count=1003,
        publish_date="2024-06-02",
        topic_label="technology",
    )
]

gold_records = [
    ArticleRecord(
        row_identifier=1,
        headline_text="Breaking: Market Rally",
        view_count=1000,
        publish_date="2024-06-01",
        topic_label="tech",
    )
]

run_config = RunConfig(
    task_type=TaskType.SINGLE_ENTITY,
    feature_rules=feature_rules,
    index_key_name="row_identifier",
)

result_bundle = evaluate(predicted_records, gold_records, run_config)
print(result_bundle.per_feature_metrics_data_frame)
```

## What this configuration demonstrates

- `headline_text` uses text normalization to ignore punctuation and case
- `view_count` allows small numeric drift
- `publish_date` allows a one-day difference
- `topic_label` uses an alias map to collapse synonyms

All four features should compare as equal in this example.

## Practical selection rules

- use `text` for free-form fields like titles or names
- use `number` when tolerances or rounding matter
- use `date` when day-window matching matters
- use `category` for canonical labels or enumerated values

## Related concepts

- [Feature Rules](../concepts/feature-rules.md)
- [Run Config](../concepts/run-config.md)
