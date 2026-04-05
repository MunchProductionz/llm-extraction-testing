# Choosing a Task Type

Choose the task type based on the shape of your records and how alignment should happen before scoring.

## Decision table

| If your data looks like this | Use this task type | Alignment strategy | `index_key_name` required | Entity presence metrics |
|---|---|---|---|---|
| One indexed record, one extracted field to compare | `SINGLE_FEATURE` | Exact join on the index key | Yes | No |
| One indexed record, multiple extracted fields to compare | `SINGLE_ENTITY` | Exact join on the index key | Yes | No |
| Lists of predicted and gold entities with no stable row key | `MULTI_ENTITY` | Entity matching, then feature scoring | No | Yes |

## Input shape examples

### `SINGLE_FEATURE`

Use this when each record contributes one label-like value.

```python
class ArticleLabel(BaseModel):
    row_identifier: int
    topic_label: str
```

Typical use cases:

- document classification labels
- one extracted category per record
- one scalar field where you want one-feature reporting

### `SINGLE_ENTITY`

Use this when each indexed record represents one entity with multiple fields.

```python
from typing import Optional
from pydantic import BaseModel


class ArticleFeatures(BaseModel):
    row_identifier: int
    headline_text: str
    author_name: str
    publish_date: Optional[str]
```

Typical use cases:

- one form or document produces one structured object
- you want per-field metrics for a single known entity per record

### `MULTI_ENTITY`

Use this when each dataset is a list of entities and the evaluator must decide which predicted entity matches which gold entity.

```python
from typing import Optional
from pydantic import BaseModel


class ContractRecord(BaseModel):
    contract_title: str
    contract_amount: Optional[float]
    contract_date: Optional[str]
```

Typical use cases:

- extracting many entities from one source document set
- invoice line items, contracts, people, products, or events
- no stable shared key exists across predicted and gold outputs

## Common mistakes

- Using `SINGLE_FEATURE` with multiple `FeatureRule` objects.
  The single-feature evaluator expects exactly one feature rule.

- Forgetting `index_key_name` for `SINGLE_FEATURE` or `SINGLE_ENTITY`.
  Both indexed task types require a join key and raise an error without one.

- Using `SINGLE_ENTITY` when the real problem is entity matching.
  If the evaluator needs to discover which entity matches which, use `MULTI_ENTITY`.

- Treating `MULTI_ENTITY` as if row order were the identity.
  Multi-entity evaluation matches entities by similarity, not by list position.

## Practical rule of thumb

Start with the simplest task type that matches your data shape:

- one field per keyed record: `SINGLE_FEATURE`
- several fields per keyed record: `SINGLE_ENTITY`
- many entities that must be matched first: `MULTI_ENTITY`

## What to read next

- [Why This Library Exists](../concepts/evaluation-driven-development.md)
- [Quickstart](quickstart.md)
- [Task Types](../concepts/task-types.md)
- [Run Config](../concepts/run-config.md)
