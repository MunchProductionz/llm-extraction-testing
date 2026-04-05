# Generate Visualizations

## Scenario

You want quick charts for a finished evaluation run so you can inspect aggregate metrics and per-feature quality visually.

## Runnable example

```python
from pydantic import BaseModel

from extraction_testing import FeatureRule, RunConfig, TaskType, evaluate
from extraction_testing.visualization import (
    plot_per_feature_metrics_bar,
    plot_total_metrics_bar,
    save_all_charts_to_report,
)


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

fig_total = plot_total_metrics_bar(result_bundle)
fig_f1 = plot_per_feature_metrics_bar(result_bundle, metric_name="f1")
paths = save_all_charts_to_report(result_bundle, "./_viz_out")

print(type(fig_total).__name__)
print(type(fig_f1).__name__)
print(paths)
```

## What to expect

- `plot_total_metrics_bar()` returns a matplotlib figure
- `plot_per_feature_metrics_bar()` returns a matplotlib figure for the chosen metric
- `save_all_charts_to_report()` creates a timestamped folder under `./_viz_out`
- the returned `paths` mapping points to the generated PNG files

## Important current caveat for multi-entity charts

`plot_entity_presence_summary()` expects summary keys named `precision`, `recall`, and `f1`, while the current multi-entity evaluator emits `precision_entities`, `recall_entities`, and `f1_entities`.

So if you want an entity-presence chart from a raw multi-entity `ResultBundle`, adapt that summary first or use a custom wrapper object.

## Related concepts

- [Visualization](../concepts/visualization.md)
- [Metrics](../concepts/metrics.md)
