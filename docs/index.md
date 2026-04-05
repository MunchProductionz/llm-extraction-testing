# extraction-testing

`extraction-testing` is a typed Python library for evaluating structured extraction outputs against gold data. It supports single-label classification-style checks, single-entity field comparison, and multi-entity matching with per-feature scoring, aggregate metrics, and optional human-readable logs.

## Supported task types

- `SINGLE_FEATURE`: evaluate one extracted feature per indexed record
- `SINGLE_ENTITY`: evaluate multiple features for one indexed entity per record
- `MULTI_ENTITY`: evaluate lists of predicted and gold entities using entity matching before scoring

## Start here

- [Quickstart](getting-started/quickstart.md)
- [Choosing a Task Type](getting-started/task-selection.md)
- [Why This Library Exists](concepts/evaluation-driven-development.md)
- [API Reference Overview](reference/overview.md)

## Why this library exists

This library is designed to make evaluation-driven development practical for extraction workflows. In many teams, building a first workflow is manageable, but building the comparison logic, metric calculations, and reporting layer around it is the part that gets skipped or delayed.

`extraction-testing` exists to standardize that evaluation layer so teams can spend more effort on the gold set, on the workflow itself, and on structured iteration with domain experts.

## Evaluation flow

1. Define one or more `FeatureRule` objects for the fields you want to compare.
2. Build a `RunConfig` with the correct `task_type` and any required alignment settings.
3. Call `evaluate(predicted_records, gold_records, run_config)` with lists of Pydantic models.
4. Inspect the returned `ResultBundle` tables and, if needed, write a text log with `RunLogger`.

## What the docs cover

- **Getting Started** shows installation, the shortest working example, and how to choose a task type.
- **Concepts** explains why the library exists and the semantics behind feature normalization, alignment, metrics, logging, and visualization.
- **How-To Guides** provide cookbook-style workflows for common evaluation tasks.
- **API Reference** maps the public package surface and will expand into module-level reference pages.
