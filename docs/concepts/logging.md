# Logging

`RunLogger` writes a human-readable text summary for one evaluation run.

## What `RunLogger` does

When you instantiate `RunLogger(log_directory_path)`, it immediately ensures the target directory exists.

When you call `write_log(...)`, it writes a file named:

```text
test_run_<run_identifier>.txt
```

The file is written inside `log_directory_path`.

## Inputs

`write_log()` takes:

- a `RunContext`
- a `ResultBundle`
- the `RunConfig`
- an optional `note_message`

The normal pattern is:

```python
from extraction_testing import RunLogger, build_run_context

run_context = build_run_context(run_config)
log_path = RunLogger(run_config.log_directory_path).write_log(
    run_context,
    result_bundle,
    run_config,
    note_message="Example run",
)
```

## What the log file contains

Current log content includes:

- run identifier
- start timestamp
- configuration hash
- task type
- total metrics table
- per-feature metrics table
- row accuracy
- entity detection summary, if present
- optional note line

## Where the `RunContext` values come from

`build_run_context(run_config)` creates:

- `run_identifier`: a timestamp-like string safe for file names
- `started_at_timestamp`: current local timestamp in ISO format
- `configuration_hash`: a deterministic short hash of the run configuration

The hash is useful when you want to compare runs with slightly different settings.

## How to interpret a log

Read it in this order:

1. confirm the `Task Type` and configuration hash match the run you intended
2. inspect total metrics for a high-level summary
3. inspect per-feature metrics to see which fields are failing
4. inspect row accuracy to understand strict full-row correctness
5. for multi-entity runs, inspect the entity summary to see whether misses are due to matching or field extraction

## Side effects

- creates the log directory if it does not already exist
- overwrites any file with the same generated name, though the timestamp-based run identifier makes collisions unlikely

## Related pages

- [Run Config](run-config.md)
- [Metrics](metrics.md)
