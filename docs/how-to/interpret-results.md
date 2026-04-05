# Interpret Results

## Scenario

You already have a `ResultBundle` and need to decide what the tables and summary fields are telling you.

## Runnable example

```python
from pydantic import BaseModel

from extraction_testing import FeatureRule, RunConfig, TaskType, evaluate


class ArticleRecord(BaseModel):
    row_identifier: int
    headline_text: str
    author_name: str


predicted_records = [
    ArticleRecord(row_identifier=1, headline_text="Market Rally", author_name="Jane Doe"),
    ArticleRecord(row_identifier=2, headline_text="Local Sports Win", author_name="J. Smith"),
]

gold_records = [
    ArticleRecord(row_identifier=1, headline_text="Market Rally", author_name="Jane Doe"),
    ArticleRecord(row_identifier=2, headline_text="Local Sports Win", author_name="John Smith"),
]

run_config = RunConfig(
    task_type=TaskType.SINGLE_ENTITY,
    feature_rules=[
        FeatureRule(feature_name="headline_text", feature_type="text"),
        FeatureRule(
            feature_name="author_name",
            feature_type="text",
            alias_map={"J. Smith": "John Smith"},
        ),
    ],
    index_key_name="row_identifier",
)

result_bundle = evaluate(predicted_records, gold_records, run_config)

print(result_bundle.per_feature_metrics_data_frame)
print(result_bundle.total_metrics_data_frame)
print("row_accuracy:", result_bundle.row_accuracy_value)
```

## How to read the result

- `per_feature_metrics_data_frame` tells you which feature is strong or weak
- `total_metrics_data_frame` gives a compact summary across features
- `row_accuracy_value` tells you how often the entire row was correct at once

In this example:

- both features should score perfectly because the author alias makes the second row correct
- `row_accuracy_value` should be `1.0` because both rows are fully correct

If you removed the alias map:

- `author_name` would weaken
- total metrics would drop
- row accuracy would also drop because the second row would no longer be fully correct

## Questions to ask when reading results

- Is the failure concentrated in one feature or spread across many?
- Is row accuracy much lower than per-feature accuracy?
- For multi-entity tasks, is the real problem entity matching rather than field extraction?

## Related concepts

- [Metrics](../concepts/metrics.md)
- [Alignment](../concepts/alignment.md)
- [Task Types](../concepts/task-types.md)
