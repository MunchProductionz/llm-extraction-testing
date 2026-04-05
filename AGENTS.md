# AGENTS.md

This file gives AI agents a concise, repository-specific guide for working in this codebase. Use it together with [`README.md`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/README.md), but treat the code as the source of truth when they differ.

## Repository purpose

`extraction-testing` is a small Python library for evaluating single-feature, single-entity, and multi-entity extraction outputs against gold data. The package is intended to become an open-source library, so changes should favor:

- a stable public API
- minimal core dependencies
- typed, predictable behavior
- clear documentation and examples

## Environment and commands

Use `uv` for local development unless a task explicitly requires something else.

### First-time setup

Core library environment:

```bash
uv sync
```

Development environment with tests and linting dependencies:

```bash
uv sync --extra dev
```

Example and demo dependencies:

```bash
uv sync --extra examples
```

### Common commands

Run tests:

```bash
uv run pytest
```

Run the synthetic end-to-end example:

```bash
uv run python examples/test.py
```

Run the visualization example:

```bash
uv run python examples/visualize_results.py
```

Run the Streamlit demo:

```bash
uv run streamlit run examples/streamlit_app.py
```

## Project layout

- [`src/extraction_testing/__init__.py`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/src/extraction_testing/__init__.py): public API surface. Preserve or intentionally evolve re-exports here.
- [`src/extraction_testing/config.py`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/src/extraction_testing/config.py): Pydantic configuration models and task enums.
- [`src/extraction_testing/orchestrator.py`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/src/extraction_testing/orchestrator.py): main entry points, especially `evaluate()` and `build_run_context()`.
- [`src/extraction_testing/tests.py`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/src/extraction_testing/tests.py): task-specific evaluator classes. Despite the filename, this is library runtime code, not pytest test code.
- [`src/extraction_testing/aligners.py`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/src/extraction_testing/aligners.py): indexed and entity alignment logic.
- [`src/extraction_testing/utils.py`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/src/extraction_testing/utils.py): normalization, equality semantics, config hashing, DataFrame conversion.
- [`src/extraction_testing/metrics.py`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/src/extraction_testing/metrics.py): per-class confusion counts, macro metrics, row accuracy, result DataFrame assembly.
- [`src/extraction_testing/models.py`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/src/extraction_testing/models.py): lightweight result dataclasses.
- [`src/extraction_testing/logger.py`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/src/extraction_testing/logger.py): human-readable log writing.
- [`src/extraction_testing/visualization.py`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/src/extraction_testing/visualization.py): optional matplotlib-based reporting helpers.
- [`tests/`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/tests): pytest suite.
- [`examples/`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/examples): runnable demos; useful for manual verification, not the import surface.

## How the library works

The main execution path is:

1. Call `evaluate(predicted_records, gold_records, run_config)`.
2. The orchestrator chooses a task-specific evaluator based on `RunConfig.task_type`.
3. Records are converted from lists of Pydantic models to pandas DataFrames.
4. Rows are aligned by either exact key match (`IndexAligner`) or weighted entity matching (`EntityAligner`), depending on the task type.
5. Feature values are normalized and compared according to each `FeatureRule`.
6. Macro metrics, micro accuracy, and row accuracy are assembled into a `ResultBundle`.
7. Optional logging and visualization operate on the `ResultBundle`.

## Important repository-specific guidance

- Keep core package dependencies small. New heavy dependencies should usually be optional extras, not mandatory runtime requirements.
- Preserve typed, deterministic behavior. Matching and normalization semantics are central to the library’s value.
- Update docs and examples when behavior changes. `README.md` is part of the package contract for future open-source users.
- Do not confuse [`src/extraction_testing/tests.py`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/src/extraction_testing/tests.py) with the pytest suite in [`tests/`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/tests).
- Generated artifacts such as `logs/` and `_viz_out/` are expected local outputs and should not be committed.

## Documentation precedence

- Treat the code as authoritative if README examples and implementation details diverge.
- Be especially careful with [`src/extraction_testing/tests.py`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/src/extraction_testing/tests.py): it contains evaluator runtime logic despite the misleading filename.

## When making changes

- Prefer small, local edits that preserve the current architecture.
- If you change public API names, also update [`src/extraction_testing/__init__.py`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/src/extraction_testing/__init__.py), [`README.md`](/Users/henrrb/Desktop/NTNU/6.klasse/Annet/Projects/llm-extraction-testing/README.md), and any affected examples.
- If you change evaluation semantics, review `utils.py`, `aligners.py`, `metrics.py`, examples, and tests together.
- If you add tests that touch visualization helpers, make sure the needed dependencies remain available through the `dev` extra so `uv sync --extra dev` is enough to run the suite.
