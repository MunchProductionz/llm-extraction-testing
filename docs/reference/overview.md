# API Reference Overview

This section documents the current public API surface exposed by `extraction_testing`.

## Public import surface

The package root currently re-exports:

- `TaskType`
- `FeatureRule`
- `MatchingConfig`
- `ClassificationConfig`
- `RunConfig`
- `ResultBundle`
- `RunContext`
- `evaluate`
- `build_run_context`
- `RunLogger`

Typical import pattern:

```python
from extraction_testing import (
    ClassificationConfig,
    FeatureRule,
    MatchingConfig,
    ResultBundle,
    RunConfig,
    RunContext,
    RunLogger,
    TaskType,
    build_run_context,
    evaluate,
)
```

## Module guide

- [config](config.md): enums and configuration models
- [orchestrator](orchestrator.md): the main evaluation entry points
- [logger](logger.md): text log writer
- [visualization](visualization.md): optional plotting helpers
- [models](models.md): result and metadata dataclasses

## What to expect from these pages

Each reference page is organized to answer the same set of questions:

- what can I import?
- what is the function or constructor signature?
- which arguments are required?
- what are the defaults and allowed values?
- what is returned?
- what side effects or error conditions should I expect?

## Notes on authority

The reference pages describe the current implementation. If an example or higher-level concept page ever disagrees with a signature or runtime behavior, the code in `src/extraction_testing/` is the source of truth.
