# Metrics

The library reports two main groups of metrics:

- feature-level metrics computed on aligned rows
- entity presence metrics for `MULTI_ENTITY` tasks

## Per-feature metrics

For each evaluated feature, the library computes:

| Metric | Meaning |
|---|---|
| `precision` | one-vs-rest precision |
| `recall` | one-vs-rest recall |
| `f1` | harmonic mean of precision and recall |
| `specificity` | one-vs-rest true negative rate |
| `micro_accuracy` | exact-match accuracy across aligned labels |

By default, these are macro-averaged across all observed labels after normalization.

## Total metrics

For multi-feature evaluators, total metrics are the arithmetic mean of the per-feature macro metrics:

- mean precision
- mean recall
- mean F1
- mean specificity

`micro_accuracy` is also included in the per-feature metrics and may appear in the total DataFrame depending on the evaluator path that produced the totals.

For `SINGLE_FEATURE`, the total metrics row is just the one feature's metrics.

## Row accuracy

`row_accuracy_value` answers a stricter question:

> Among aligned rows, how often were all evaluated features correct at the same time?

Current behavior:

- for multi-feature tasks, the evaluator builds a boolean equality matrix and requires every feature in a row to be `True`
- for `SINGLE_FEATURE`, row accuracy reduces to exact label agreement on aligned rows
- if there are no aligned rows, row accuracy is `0.0`

## Entity presence metrics

`MULTI_ENTITY` evaluation also produces `entity_detection_summary` with:

- `predicted_count`
- `gold_count`
- `matched_count`
- `extra_predictions_count`
- `missed_gold_count`
- `precision_entities`
- `recall_entities`
- `f1_entities`

These metrics are based on matched entities, not on field values inside the entities.

## Macro vs binary reporting

The main metric function behaves in two modes:

- if `classification_config.positive_label` is `None`, return macro precision/recall/F1/specificity across observed labels
- if `positive_label` is set and appears in the label set, return one-vs-rest metrics for that label

In both cases, `micro_accuracy` remains the fraction of aligned labels that match exactly.

## Missing-value semantics

Missing-value handling depends on the feature type.

### Text and category

Current implementation behavior:

- values are converted to strings before normalization
- `None` therefore becomes `"none"` after casefolding
- no special missing-value equality rule is applied

### Number

Current implementation behavior:

- `None`, `NaN`, infinite values, and unparsable numbers are treated as missing
- both missing means equal
- one missing and one present means unequal

### Date

Current implementation behavior:

- unparsable dates are treated as missing
- both missing means equal
- one missing and one present means unequal

## Important scope note for `MULTI_ENTITY`

Per-feature metrics and row accuracy are computed only on matched entity pairs.

That means:

- a system can have strong per-feature scores on matched entities
- while still having poor entity presence recall because many gold entities were never matched

Use both metric groups together when evaluating multi-entity extraction quality.

## Related pages

- [Task Types](task-types.md)
- [Feature Rules](feature-rules.md)
- [Alignment](alignment.md)
