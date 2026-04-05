# Evaluate a Multi-Entity Task

## Scenario

You have lists of predicted and gold entities and there is no stable shared key between them. The evaluator must first decide which predicted entity matches which gold entity.

## Runnable example

```python
from typing import Optional

from pydantic import BaseModel

from extraction_testing.config import MatchingConfig
from extraction_testing import FeatureRule, RunConfig, TaskType, evaluate


class ContractRecord(BaseModel):
    contract_title: str
    contract_amount: Optional[float]


predicted_records = [
    ContractRecord(contract_title="Supply Agreement Alpha", contract_amount=100000.0),
    ContractRecord(contract_title="Maintenance Contract Beta", contract_amount=50520.0),
    ContractRecord(contract_title="Acquisition Accord Zeta", contract_amount=300000.0),
]

gold_records = [
    ContractRecord(contract_title="Supply Agreement Alpha", contract_amount=100000.0),
    ContractRecord(contract_title="Maintenance Contract Beta", contract_amount=50500.0),
    ContractRecord(contract_title="Licensing Agreement Epsilon", contract_amount=150000.0),
]

run_config = RunConfig(
    task_type=TaskType.MULTI_ENTITY,
    feature_rules=[
        FeatureRule(feature_name="contract_title", feature_type="text", weight_for_matching=2.0),
        FeatureRule(
            feature_name="contract_amount",
            feature_type="number",
            numeric_absolute_tolerance=100.0,
            weight_for_matching=1.0,
        ),
    ],
    matching_config=MatchingConfig(
        matching_mode="weighted",
        minimum_similarity_threshold=0.6,
    ),
)

result_bundle = evaluate(predicted_records, gold_records, run_config)

print(result_bundle.per_feature_metrics_data_frame)
print(result_bundle.total_metrics_data_frame)
print("row_accuracy:", result_bundle.row_accuracy_value)
print(result_bundle.entity_detection_summary)
print(result_bundle.matched_pairs_data_frame)
```

## What to expect

- the first two predicted entities should match the first two gold entities
- the third predicted entity should remain unmatched
- the third gold entity should remain missed

So the entity summary should show:

- `matched_count` of `2`
- one extra prediction
- one missed gold entity
- `precision_entities` and `recall_entities` both around `0.6667`

The matched rows should still score well at the feature level because the amount difference for the beta contract is within the configured tolerance.

## When this guide applies

Use this workflow when:

- entity identity must be inferred from content
- you need both feature-level quality and entity-detection quality
- row order should not be treated as identity

## Related concepts

- [Alignment](../concepts/alignment.md)
- [Feature Rules](../concepts/feature-rules.md)
- [Metrics](../concepts/metrics.md)
