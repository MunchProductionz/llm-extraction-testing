# Feature Rules

A `FeatureRule` defines how one feature should be compared. The same rule affects both equality checks for scoring and similarity scoring for entity matching.

## Field reference

| Field | Type | Default | Used for | Notes |
|---|---|---|---|---|
| `feature_name` | `str` | required | all tasks | Must match the model/DataFrame column name |
| `feature_type` | `str` | required | all tasks | Must be one of `text`, `number`, `date`, `category` |
| `is_mandatory_for_matching` | `bool` | `True` | exact entity matching | If any mandatory feature differs, exact matching returns similarity `0.0` |
| `weight_for_matching` | `float` | `1.0` | weighted entity matching | Controls contribution to weighted similarity |
| `casefold_text` | `bool` | `True` | text/category | Case-insensitive normalization |
| `strip_text` | `bool` | `True` | text/category | Trims leading and trailing whitespace |
| `remove_punctuation` | `bool` | `True` | text/category | Removes punctuation before comparison |
| `alias_map` | `dict[str, str] \| None` | `None` | text/category, matching | Applied before normalization |
| `numeric_rounding_digits` | `int \| None` | `None` | number | Rounds parsed numeric values before equality |
| `numeric_absolute_tolerance` | `float \| None` | `None` | number | Accepts values within absolute difference |
| `numeric_relative_tolerance` | `float \| None` | `None` | number | Accepts values within relative difference when gold is nonzero |
| `date_tolerance_days` | `int \| None` | `None` | date | Accepts dates within a day window |

## Allowed `feature_type` values

`feature_type` is the only field with built-in validation in the model. The current allowed values are:

- `text`
- `number`
- `date`
- `category`

Any other value raises `ValueError` during model construction.

## Text and category semantics

For `text` and `category` features, the library:

1. applies `alias_map` if provided
2. converts the value to a string
3. casefolds, strips, removes punctuation, and collapses repeated whitespace
4. compares the normalized strings

Example:

```python
FeatureRule(
    feature_name="currency_code",
    feature_type="category",
    alias_map={"US Dollar": "USD"},
)
```

With the default text settings, `"US Dollar"` and `"USD"` compare as equal after aliasing.

### Current implementation note for missing text values

The current runtime path stringifies text and category values before normalization. That means `None` is treated like the string `"None"` during comparison, not like a dedicated missing-value sentinel.

This differs from the number and date behavior, which handle missing values explicitly.

## Number semantics

For `number` features, the library:

1. tries to parse each value as `float`
2. optionally rounds the parsed values
3. treats both-missing as equal
4. treats one-missing and one-present as unequal
5. checks absolute tolerance, then relative tolerance, then exact numeric equality

Example:

```python
FeatureRule(
    feature_name="contract_amount",
    feature_type="number",
    numeric_rounding_digits=0,
    numeric_absolute_tolerance=100.0,
)
```

With this rule, `100000.0` and `100049.0` compare as equal.

### Missing and unparsable numeric values

- `None` stays missing
- `NaN` and infinite floats are treated as missing
- unparsable values such as `""` are treated as missing

## Date semantics

For `date` features, the library:

1. parses values with `datetime.fromisoformat(...).date()`
2. treats both-missing as equal
3. treats one-missing and one-present as unequal
4. applies `date_tolerance_days` if configured
5. otherwise requires exact date equality

Example:

```python
FeatureRule(
    feature_name="publish_date",
    feature_type="date",
    date_tolerance_days=1,
)
```

With this rule, `2024-06-01` and `2024-06-02` compare as equal.

## Matching behavior

`FeatureRule` also affects multi-entity matching:

- in `matching_mode="exact"`, only `is_mandatory_for_matching` matters
- in `matching_mode="weighted"`, every feature contributes a similarity score multiplied by `weight_for_matching`
- text features get partial similarity through token-set overlap
- category, number, and date features contribute either `1.0` or `0.0`

## Recommended usage

- Use `alias_map` for known synonyms or canonical value mapping.
- Use tolerances for values where small drift is acceptable.
- Increase `weight_for_matching` on features that are especially identifying in `MULTI_ENTITY` tasks.
- Keep feature names aligned exactly with your Pydantic model fields.

## Related pages

- [Run Config](run-config.md)
- [Alignment](alignment.md)
- [Metrics](metrics.md)
