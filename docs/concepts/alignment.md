# Alignment

Alignment decides which predicted rows are compared against which gold rows. The library has two alignment strategies: indexed matching and entity matching.

## `IndexAligner`

`IndexAligner` is used for `SINGLE_FEATURE` and `SINGLE_ENTITY`.

Current behavior:

- reads `run_config.index_key_name`
- builds a map from each gold key value to one gold row index
- iterates through predicted rows and matches rows with the same key
- returns matched pairs with similarity score `1.0`
- also returns unmatched predicted indices and unmatched gold indices

### Important implications

- matching is exact equality on the key field
- duplicate key values in the gold DataFrame are ambiguous and should be avoided
- unmatched rows are not included in per-feature scoring

## `EntityAligner`

`EntityAligner` is used for `MULTI_ENTITY`.

Current behavior:

1. build candidate predicted/gold pairs
2. compute similarity for each pair
3. drop pairs below `minimum_similarity_threshold`
4. optionally keep only the top `maximum_candidate_pairs`
5. shuffle candidates using a fixed seed
6. stable-sort by similarity descending
7. greedily choose non-overlapping pairs

The result is deterministic for a fixed seed and input order.

## Matching modes

### Exact matching

When `matching_mode="exact"`:

- the aligner checks only features where `is_mandatory_for_matching=True`
- if any mandatory feature is unequal, pair similarity is `0.0`
- otherwise pair similarity is `1.0`

This is strict matching. There is no partial credit.

### Weighted matching

When `matching_mode="weighted"`:

- every feature contributes a similarity score
- the final score is a weighted average
- weights come from `FeatureRule.weight_for_matching`

Feature-level similarity rules:

- `text`: token-set overlap after normalization
- `category`: `1.0` if equal, else `0.0`
- `number`: `1.0` if equal under tolerance rules, else `0.0`
- `date`: `1.0` if equal under date tolerance rules, else `0.0`

## Threshold behavior

`minimum_similarity_threshold` applies after pair similarity is computed.

- pairs below the threshold are ignored entirely
- pairs at or above the threshold remain candidates for greedy selection

Practical effect:

- a high threshold increases precision of entity matching
- a low threshold increases recall of candidate pairs but can create more ambiguous competition

## Determinism and tie-breaking

For weighted matching, the aligner:

- seeds Python's `random` module with `random_tie_breaker_seed`
- shuffles candidate pairs
- then sorts by similarity descending using a stable sort

This means equal-score pairs are resolved reproducibly for a given seed.

## What alignment affects downstream

- matched pairs feed the per-feature metrics and row accuracy calculations
- unmatched predicted and unmatched gold entities feed the multi-entity presence summary
- indexed tasks do not expose entity presence metrics because alignment is assumed to be explicit

## Related pages

- [Task Types](task-types.md)
- [Feature Rules](feature-rules.md)
- [Metrics](metrics.md)
