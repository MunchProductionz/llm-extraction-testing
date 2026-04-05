# Documentation Progress Tracker

This file is the mutable work tracker for documentation work.

Use `DOCS_ARCHITECTURE_PLAN.md` for the stable design. Use this file for:

- current status
- ownership
- handoffs
- blockers
- next actions

## Update Rules

Every AI agent that works on docs should update this file before finishing.

Do not rewrite history. Append updates.

When updating this file:

- move task status forward only if the work is actually complete
- leave concise notes about what changed
- list blockers explicitly
- add follow-up tasks instead of burying them in prose

## Status Legend

- `pending`: not started
- `in_progress`: currently being worked on
- `blocked`: cannot proceed without a decision or dependency
- `done`: completed and reviewed enough for handoff

## Current Snapshot

Last updated: 2026-04-05

Overall docs program status: `planning_complete`

Current state:

- planning docs exist
- information architecture is defined
- page tree is defined
- MkDocs scaffolding and nav skeleton exist
- Getting Started pages have been written
- Concepts pages have been written
- How-to guides have been written
- API reference pages have been written
- contributor docs have been written
- GitHub Pages publishing workflow has been added
- strict MkDocs build has passed during final QA
- motivation and EDD docs have been added

## Work Packages

| ID | Work package | Status | Owner | Depends on | Notes |
|---|---|---|---|---|---|
| WP-00 | Planning artifacts | done | current agent | none | `DOCS_ARCHITECTURE_PLAN.md` and `DOCS_PROGRESS.md` created |
| WP-01 | MkDocs site scaffolding | done | Codex | WP-00 | Added `mkdocs.yml`, docs extra, nav skeleton, and placeholder pages for the planned tree |
| WP-02 | Getting Started pages | done | Codex | WP-01 | Wrote home, installation, quickstart, and task-selection pages with task-first examples |
| WP-03 | Concepts pages | done | Codex | WP-01 | Wrote all concept pages from current runtime behavior, including validation gaps and visualization caveats |
| WP-04 | API reference pages | done | Codex | WP-01 | Added reference overview pages with hand-authored summaries plus mkdocstrings directives |
| WP-05 | How-to guides | done | Codex | WP-02, WP-03 | Wrote runnable task and workflow guides with expected-result notes |
| WP-06 | Contributor docs | done | Codex | WP-00 | Wrote docs style guide, maintenance guide, and AI handoff guide |
| WP-07 | Docs publishing setup | done | Codex | WP-01 | Added GitHub Pages deployment workflow for MkDocs |
| WP-08 | Final docs QA pass | done | Codex | WP-02, WP-03, WP-04, WP-05, WP-06 | Placeholder scan, diff check, and strict MkDocs build passed |

## Page-Level Backlog

Use this section to claim specific pages if work is split more finely than work packages.

| Page | Status | Owner | Notes |
|---|---|---|---|
| `docs/index.md` | done | Codex | Home page and summary written |
| `docs/getting-started/installation.md` | done | Codex | `uv`-first setup, extras, tests, and docs commands included |
| `docs/getting-started/quickstart.md` | done | Codex | Minimal `SINGLE_FEATURE` example plus optional logging step |
| `docs/getting-started/task-selection.md` | done | Codex | Decision table, shape examples, and common mistakes |
| `docs/concepts/task-types.md` | done | Codex | Task semantics, aligners, and metric scope documented |
| `docs/concepts/feature-rules.md` | done | Codex | Field-by-field semantics, defaults, and missing-value behavior documented |
| `docs/concepts/run-config.md` | done | Codex | Task-dependent requirements and currently unused fields documented |
| `docs/concepts/alignment.md` | done | Codex | Indexed vs weighted matching, thresholds, and determinism documented |
| `docs/concepts/metrics.md` | done | Codex | Macro, binary, row, entity, and missing-value semantics documented |
| `docs/concepts/logging.md` | done | Codex | `RunLogger` inputs, outputs, and side effects documented |
| `docs/concepts/visualization.md` | done | Codex | Plot functions, report output, and current implementation caveats documented |
| `docs/concepts/evaluation-driven-development.md` | done | Codex | Motivation page explaining why the library exists and how it fits EDD |
| `docs/how-to/evaluate-single-feature.md` | done | Codex | Runnable example with expected row accuracy |
| `docs/how-to/evaluate-single-entity.md` | done | Codex | Runnable example with field-level interpretation |
| `docs/how-to/evaluate-multi-entity.md` | done | Codex | Runnable example with entity-summary interpretation |
| `docs/how-to/configure-feature-rules.md` | done | Codex | Complete feature-rule setup example across feature types |
| `docs/how-to/interpret-results.md` | done | Codex | Result-bundle reading guide with comparison example |
| `docs/how-to/save-logs.md` | done | Codex | End-to-end logging workflow |
| `docs/how-to/generate-visualizations.md` | done | Codex | Plot creation and report-saving workflow |
| `docs/reference/overview.md` | done | Codex | Public API map and import surface documented |
| `docs/reference/config.md` | done | Codex | Config models and enums documented with mkdocstrings hook |
| `docs/reference/orchestrator.md` | done | Codex | `evaluate` and `build_run_context` documented |
| `docs/reference/logger.md` | done | Codex | `RunLogger` documented |
| `docs/reference/visualization.md` | done | Codex | Plot/save functions documented with caveats |
| `docs/reference/models.md` | done | Codex | Result dataclasses documented |
| `docs/contributor/docs-style-guide.md` | done | Codex | Writing conventions and docs-quality rules documented |
| `docs/contributor/docs-maintenance.md` | done | Codex | Local docs workflow, sync points, and QA checklist documented |
| `docs/contributor/ai-handoff.md` | done | Codex | Agent workflow for claiming, verifying, and handing off docs work documented |

## Current Blockers

No current blocker.

## Recommended Next Steps

Recommended maintenance and enhancement work:

1. keep README and docs pages aligned when evaluation semantics change
2. keep reference-page summaries aligned with generated API details and docstrings
3. consider adding stronger automated docs QA beyond `mkdocs build --strict` if link validation becomes important

## Handoff Log

Append new entries at the bottom.

### 2026-04-05 - current agent

- Completed: created planning artifacts for multi-agent documentation work
- Files added:
  - `docs/planning/DOCS_ARCHITECTURE_PLAN.md`
  - `docs/planning/DOCS_PROGRESS.md`
- Outcome:
  - defined documentation information architecture
  - defined exact proposed MkDocs page tree
  - split documentation work into work packages
  - established update protocol for future AI agents
- Recommended next task:
  - implement WP-01 and create the actual MkDocs scaffolding

### 2026-04-05 - Codex

- Completed:
  - implemented WP-01 MkDocs scaffolding
  - implemented WP-02 Getting Started pages
- Files added/updated:
  - `mkdocs.yml`
  - `pyproject.toml`
  - `docs/index.md`
  - `docs/getting-started/installation.md`
  - `docs/getting-started/quickstart.md`
  - `docs/getting-started/task-selection.md`
  - placeholder pages across `docs/concepts/`, `docs/how-to/`, `docs/reference/`, and `docs/contributor/`
  - `docs/planning/DOCS_PROGRESS.md`
- Work package status changes:
  - `WP-01` -> `done`
  - `WP-02` -> `done`
- Blockers:
  - no technical blocker for writing the next docs pages
  - API reference strategy is still only partially resolved; current reference pages are placeholders until WP-04
- Follow-up recommendations:
  - write WP-03 Concepts pages next, starting with `task-types`, `feature-rules`, and `run-config`
  - then replace API reference placeholders with mkdocstrings-backed pages in WP-04

### 2026-04-05 - Codex

- Completed:
  - implemented WP-03 Concepts pages
- Files added/updated:
  - `docs/concepts/task-types.md`
  - `docs/concepts/feature-rules.md`
  - `docs/concepts/run-config.md`
  - `docs/concepts/alignment.md`
  - `docs/concepts/metrics.md`
  - `docs/concepts/logging.md`
  - `docs/concepts/visualization.md`
  - `docs/planning/DOCS_PROGRESS.md`
- Work package status changes:
  - `WP-03` -> `done`
- Blockers:
  - no blocker for continuing into how-to or reference docs
  - some current runtime behaviors are now documented as-is and may warrant future code cleanup:
    - text/category `None` handling is string-based
    - `grouping_key_names` and `average_strategy` are present but unused
    - visualization entity-summary key expectations do not match evaluator output keys
- Follow-up recommendations:
  - write WP-05 how-to guides next while the concept pages are fresh
  - handle WP-04 after deciding how much of the API reference should be generated with mkdocstrings vs hand-authored

### 2026-04-05 - Codex

- Completed:
  - implemented WP-05 How-to guides
  - implemented WP-04 API reference pages
- Files added/updated:
  - `docs/how-to/evaluate-single-feature.md`
  - `docs/how-to/evaluate-single-entity.md`
  - `docs/how-to/evaluate-multi-entity.md`
  - `docs/how-to/configure-feature-rules.md`
  - `docs/how-to/interpret-results.md`
  - `docs/how-to/save-logs.md`
  - `docs/how-to/generate-visualizations.md`
  - `docs/reference/overview.md`
  - `docs/reference/config.md`
  - `docs/reference/orchestrator.md`
  - `docs/reference/logger.md`
  - `docs/reference/visualization.md`
  - `docs/reference/models.md`
  - `docs/planning/DOCS_PROGRESS.md`
- Work package status changes:
  - `WP-05` -> `done`
  - `WP-04` -> `done`
- Blockers:
  - no blocker for continuing to contributor docs or publishing setup
  - mkdocstrings-backed pages have not been validated with a full docs build in this turn
- Follow-up recommendations:
  - write WP-06 contributor docs next
  - then run a docs build / QA pass once the remaining sections are in place

### 2026-04-05 - Codex

- Completed:
  - implemented WP-06 contributor docs
  - implemented WP-07 docs publishing setup
  - implemented WP-08 final docs QA pass
- Files added/updated:
  - `docs/contributor/docs-style-guide.md`
  - `docs/contributor/docs-maintenance.md`
  - `docs/contributor/ai-handoff.md`
  - `mkdocs.yml`
  - `.github/workflows/docs.yml`
  - `docs/planning/DOCS_PROGRESS.md`
- Work package status changes:
  - `WP-06` -> `done`
  - `WP-07` -> `done`
  - `WP-08` -> `done`
- Blockers:
  - no remaining blocker for the planned docs rollout
  - note: the final build showed an upstream Material for MkDocs warning about MkDocs 2.0; it did not block the current build
- Follow-up recommendations:
  - if you want tighter CI validation later, consider adding explicit link-validation settings or a dedicated docs QA workflow step beyond `mkdocs build --strict`

### 2026-04-05 - Codex

- Completed:
  - added motivation content to the README and docs home page
  - added a dedicated EDD motivation concepts page
  - added Mermaid support for a documentation workflow diagram
  - updated adjacent docs and planning files for consistency
- Files added/updated:
  - `README.md`
  - `mkdocs.yml`
  - `pyproject.toml`
  - `docs/index.md`
  - `docs/getting-started/task-selection.md`
  - `docs/concepts/task-types.md`
  - `docs/concepts/evaluation-driven-development.md`
  - `docs/javascripts/mermaid-init.js`
  - `docs/planning/DOCS_ARCHITECTURE_PLAN.md`
  - `docs/planning/DOCS_PROGRESS.md`
- Work package status changes:
  - no existing work package changed; this was a post-rollout documentation enhancement
- Blockers:
  - no blocker
  - note: the strict build still shows the existing upstream Material for MkDocs warning about MkDocs 2.0, but the build passes
- Follow-up recommendations:
  - if you add more high-level conceptual pages later, keep the README summary shorter than the dedicated docs page and let the concepts section carry the full explanation

## Handoff Entry Template

Copy this section for future updates:

```text
### YYYY-MM-DD - <agent label>

- Completed:
- Files added/updated:
- Work package status changes:
- Blockers:
- Follow-up recommendations:
```
