# Documentation Maintenance

This page describes the ongoing maintenance loop for the docs site and how to keep it aligned with the library code.

## Local docs commands

Install docs dependencies:

```bash
uv sync --extra docs
```

Serve the docs locally:

```bash
uv run mkdocs serve
```

Build the docs site:

```bash
uv run mkdocs build --strict
```

## When docs must be updated

Update docs whenever a change affects:

- public API names or signatures
- task semantics or evaluation behavior
- feature normalization or matching behavior
- logging output or visualization behavior
- installation or development workflow

At minimum, review:

- `README.md`
- the relevant page under `docs/`
- any examples affected by the change
- `docs/planning/DOCS_PROGRESS.md` if the change is documentation work

## Source-of-truth rule

When docs and code disagree:

- code wins
- docs should be updated to match the current implementation
- if the implementation is wrong, fix the code and the docs together

Primary implementation sources:

- `src/extraction_testing/config.py`
- `src/extraction_testing/orchestrator.py`
- `src/extraction_testing/tests.py`
- `src/extraction_testing/aligners.py`
- `src/extraction_testing/utils.py`
- `src/extraction_testing/metrics.py`
- `src/extraction_testing/logger.py`
- `src/extraction_testing/models.py`
- `src/extraction_testing/visualization.py`

## Documentation maintenance checklist

For any meaningful library change:

1. update the relevant docs pages
2. update any examples that demonstrate the changed behavior
3. review the reference pages for signature or default drift
4. run `uv run mkdocs build --strict`
5. if available, run the relevant tests or examples

## Publishing workflow

The repository uses a GitHub Actions workflow for docs publishing:

- workflow file: `.github/workflows/docs.yml`
- build command: `uv run mkdocs build --strict`
- deployment target: GitHub Pages

The deploy job is intended for the main branch, while pull requests should still build the docs to catch regressions before merge.

## Common failure modes

- page exists in `docs/` but is missing from `mkdocs.yml`
- links point to outdated page paths
- examples use stale enum names or constructor fields
- docs describe planned behavior instead of current behavior
- generated API pages drift because supporting hand-authored summaries were not updated

## Final QA expectations

Before considering docs work complete, try to verify:

- pages exist for all nav entries
- links resolve
- names and terminology are consistent
- examples match current code
- there is no leftover placeholder text

## Related pages

- [Documentation Style Guide](docs-style-guide.md)
- [AI Handoff](ai-handoff.md)
