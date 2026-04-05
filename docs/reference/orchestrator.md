# `orchestrator`

This module contains the main user-facing entry points for running evaluations.

## `evaluate(predicted_records, gold_records, run_config)`

Purpose:

- choose the correct evaluator class from `run_config.task_type`
- run the evaluation
- return a `ResultBundle`

Parameters:

- `predicted_records`: list of Pydantic models representing predictions
- `gold_records`: list of Pydantic models representing gold data
- `run_config`: `RunConfig` controlling task type and comparison behavior

Returns:

- `ResultBundle`

Error conditions:

- unsupported `task_type` raises `ValueError`
- indexed task types raise `ValueError` if `index_key_name` is missing
- single-feature evaluation raises `ValueError` if more than one feature rule is supplied

Side effects:

- none

## `build_run_context(run_config)`

Purpose:

- create a `RunContext` for logging and traceability

Returns:

- `RunContext` containing a run identifier, start timestamp, and configuration hash

Side effects:

- none

## Generated API details

::: extraction_testing.orchestrator
    options:
      members:
        - build_run_context
        - evaluate
      show_root_heading: false
