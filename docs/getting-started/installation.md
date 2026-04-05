# Installation

Use `uv` for local development unless you have a reason to use `pip` directly.

## Core library setup

Install the package and its core runtime dependencies:

```bash
uv sync
```

This gives you the library with the dependencies needed to run evaluations.

## Development setup

Install the package with test, lint, and type-check tooling:

```bash
uv sync --extra dev
```

Run the test suite:

```bash
uv run pytest
```

## Example and visualization dependencies

Install optional dependencies used by the example scripts and Streamlit demo:

```bash
uv sync --extra examples
```

Example commands:

```bash
uv run python examples/test.py
uv run python examples/visualize_results.py
uv run streamlit run examples/streamlit_app.py
```

## Documentation dependencies

Install the documentation toolchain:

```bash
uv sync --extra docs
```

Serve the docs site locally:

```bash
uv run mkdocs serve
```

## `pip` alternative

If you are not using `uv`, install the package in editable mode:

```bash
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

## What to read next

- [Quickstart](quickstart.md) for the smallest realistic evaluation
- [Choosing a Task Type](task-selection.md) if you are not sure which evaluator shape matches your data
