# Task Types

`extraction-testing` supports three task types. They differ mainly in how records are aligned before feature-level comparison starts.

## Overview

| Task type | Intended shape | Alignment strategy | `index_key_name` | Entity presence metrics |
|---|---|---|---|---|
| `SINGLE_FEATURE` | One feature per indexed record | `IndexAligner` | Required | No |
| `SINGLE_ENTITY` | Multiple features per indexed record | `IndexAligner` | Required | No |
| `MULTI_ENTITY` | Lists of entities that must be matched | `EntityAligner` | Not required | Yes |

## `SINGLE_FEATURE`

Use `SINGLE_FEATURE` when each record has one value you want to score, such as a topic label or one extracted category.

Current runtime behavior:

- records are aligned by exact equality on `RunConfig.index_key_name`
- exactly one `FeatureRule` must be provided
- metrics are computed for that one feature
- `row_accuracy_value` is the same idea as exact-label accuracy for the aligned rows

This is the simplest evaluator shape and usually the best starting point for classification-style outputs.

## `SINGLE_ENTITY`

Use `SINGLE_ENTITY` when each indexed record represents one entity with several fields, such as one article with headline, author, and publish date.

Current runtime behavior:

- records are aligned by exact equality on `RunConfig.index_key_name`
- one or more `FeatureRule` objects can be provided
- per-feature metrics are computed after alignment
- `row_accuracy_value` is the fraction of aligned rows where every tested feature matches

This task type assumes the identity of the entity is already known through the index key.

## `MULTI_ENTITY`

Use `MULTI_ENTITY` when predicted and gold data are both lists of entities and the evaluator must decide which predicted entity matches which gold entity.

Current runtime behavior:

- alignment is handled by `EntityAligner`
- `RunConfig.matching_config` controls exact or weighted matching behavior
- per-feature metrics are computed only on matched pairs
- unmatched predicted and unmatched gold entities are summarized separately in `entity_detection_summary`

This matters because feature metrics and entity presence metrics answer different questions:

- feature metrics describe how accurate matched entity fields are
- entity presence metrics describe whether the system found the right entities at all

## Which aligner each task uses

- `SINGLE_FEATURE` and `SINGLE_ENTITY` use [`IndexAligner`](../concepts/alignment.md), which joins rows using the configured key.
- `MULTI_ENTITY` uses [`EntityAligner`](../concepts/alignment.md), which scores candidate pairs and then chooses non-overlapping matches.

## Practical implications

- If row identity is already known, prefer an indexed task type.
- If row identity must be inferred from content similarity, use `MULTI_ENTITY`.
- If you only care about one output field, use `SINGLE_FEATURE` so the result tables stay minimal.

## Related pages

- [Why This Library Exists](evaluation-driven-development.md)
- [Choosing a Task Type](../getting-started/task-selection.md)
- [Run Config](run-config.md)
- [Alignment](alignment.md)
