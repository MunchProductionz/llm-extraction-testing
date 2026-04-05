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
- no browser docs tooling has been added yet
- no end-user docs pages have been written yet

## Work Packages

| ID | Work package | Status | Owner | Depends on | Notes |
|---|---|---|---|---|---|
| WP-00 | Planning artifacts | done | current agent | none | `DOCS_ARCHITECTURE_PLAN.md` and `DOCS_PROGRESS.md` created |
| WP-01 | MkDocs site scaffolding | pending | unassigned | WP-00 | Add `mkdocs.yml`, docs plugin/theme config, and nav skeleton |
| WP-02 | Getting Started pages | pending | unassigned | WP-01 | Installation, quickstart, task selection |
| WP-03 | Concepts pages | pending | unassigned | WP-01 | Task types, feature rules, run config, alignment, metrics, logging, visualization |
| WP-04 | API reference pages | pending | unassigned | WP-01 | Prefer generated reference with hand-written overview |
| WP-05 | How-to guides | pending | unassigned | WP-02, WP-03 | One workflow page per common task |
| WP-06 | Contributor docs | pending | unassigned | WP-00 | Docs style guide, maintenance guide, AI handoff guide |
| WP-07 | Docs publishing setup | pending | unassigned | WP-01 | GitHub Pages or equivalent deployment workflow |
| WP-08 | Final docs QA pass | pending | unassigned | WP-02, WP-03, WP-04, WP-05, WP-06 | Link check, naming consistency, examples sanity check |

## Page-Level Backlog

Use this section to claim specific pages if work is split more finely than work packages.

| Page | Status | Owner | Notes |
|---|---|---|---|
| `docs/index.md` | pending | unassigned | Home page and summary |
| `docs/getting-started/installation.md` | pending | unassigned | `uv`-first setup |
| `docs/getting-started/quickstart.md` | pending | unassigned | Minimal end-to-end example |
| `docs/getting-started/task-selection.md` | pending | unassigned | Decision table for task types |
| `docs/concepts/task-types.md` | pending | unassigned | Must align exactly with current enum names |
| `docs/concepts/feature-rules.md` | pending | unassigned | Full field-by-field semantics |
| `docs/concepts/run-config.md` | pending | unassigned | Task-dependent requirements |
| `docs/concepts/alignment.md` | pending | unassigned | Indexed vs weighted matching |
| `docs/concepts/metrics.md` | pending | unassigned | Macro, micro, row, entity metrics |
| `docs/concepts/logging.md` | pending | unassigned | `RunLogger` behavior |
| `docs/concepts/visualization.md` | pending | unassigned | Plot functions and report generation |
| `docs/how-to/evaluate-single-feature.md` | pending | unassigned | Runnable example |
| `docs/how-to/evaluate-single-entity.md` | pending | unassigned | Runnable example |
| `docs/how-to/evaluate-multi-entity.md` | pending | unassigned | Runnable example |
| `docs/how-to/configure-feature-rules.md` | pending | unassigned | Text/number/date/category examples |
| `docs/how-to/interpret-results.md` | pending | unassigned | Explain result bundle tables and metrics |
| `docs/how-to/save-logs.md` | pending | unassigned | Logging workflow |
| `docs/how-to/generate-visualizations.md` | pending | unassigned | Chart generation workflow |
| `docs/reference/overview.md` | pending | unassigned | Public API map |
| `docs/reference/config.md` | pending | unassigned | Config models and enums |
| `docs/reference/orchestrator.md` | pending | unassigned | `evaluate`, `build_run_context` |
| `docs/reference/logger.md` | pending | unassigned | `RunLogger` |
| `docs/reference/visualization.md` | pending | unassigned | Plot/save functions |
| `docs/reference/models.md` | pending | unassigned | `ResultBundle`, `RunContext` |
| `docs/contributor/docs-style-guide.md` | pending | unassigned | Writing conventions |
| `docs/contributor/docs-maintenance.md` | pending | unassigned | How to keep docs in sync |
| `docs/contributor/ai-handoff.md` | pending | unassigned | Agent-facing quick handoff guide |

## Current Blockers

No blocking technical issue yet.

Open design decisions that may block later work:

- generated vs hand-authored API reference details
- exact docs theme/tooling choice
- docs deployment workflow

## Recommended Next Steps

Recommended implementation order:

1. WP-01: set up MkDocs scaffolding and create empty page files
2. WP-02: write Getting Started pages
3. WP-03: write Concepts pages
4. WP-04: build API reference pages
5. WP-05: add How-to guides
6. WP-06: add Contributor docs
7. WP-07: wire browser publishing
8. WP-08: final QA pass

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
