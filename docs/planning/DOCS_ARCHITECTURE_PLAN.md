# Documentation Architecture Plan

This document is the source of truth for how the documentation for `extraction-testing` should be designed and built.

It is written for both:

- human maintainers deciding what documentation should exist
- AI agents implementing parts of the documentation over multiple handoffs

This file should change rarely. Use `DOCS_PROGRESS.md` for status updates, ownership, and handoff notes.

## Purpose

The documentation should make it easy for a user to answer four questions:

1. What task types does this library support?
2. How do I run an evaluation for my task?
3. How do `FeatureRule`, `RunConfig`, `RunLogger`, and the visualization helpers work?
4. What are the exact arguments, allowed values, defaults, return values, and side effects of the public API?

The documentation should also be usable by AI agents. That means:

- documentation source must live in the repository
- page structure must be explicit and predictable
- authoritative source files must be identified
- unfinished sections must be easy to discover
- handoff notes must be tracked separately from the architecture plan

## Current State

As of 2026-04-05, the repository has:

- `README.md` with installation, quickstart, metrics overview, visualization notes, and motivation
- `AGENTS.md` with repository guidance for AI agents
- a structured MkDocs documentation site in `docs/`
- concept, how-to, reference, contributor, and planning pages
- docstrings in the Python source

This means future work should focus on maintaining alignment between the code and the existing docs structure, not on inventing a new documentation layout.

## Recommended Tooling

The recommended path is:

- MkDocs
- Material for MkDocs
- mkdocstrings for Python API reference generation

Rationale:

- static Markdown docs are easy for humans and AI agents to edit
- the docs can be rendered locally and hosted from GitHub Pages
- the information architecture is easy to keep in sync with repository files
- generated API reference pages reduce drift for function/class signatures

The docs should follow the same broad pattern used by DocETL:

- overview and getting started
- concept docs
- how-to guides
- API reference
- best-practices / design notes when useful

## Documentation Principles

All future docs should follow these principles:

- Task-first before module-first. Lead with `SINGLE_FEATURE`, `SINGLE_ENTITY`, and `MULTI_ENTITY`.
- Concepts before raw API. Explain alignment, normalization, and metrics before listing parameters.
- API reference must be exhaustive. Every public constructor/function should document arguments, defaults, allowed values, returns, and exceptions or failure modes.
- Avoid hidden behavior. Optional-field semantics, tolerances, aliasing, and matching behavior must be documented explicitly.
- Keep one source of truth per concern. Stable architecture belongs here; mutable work tracking belongs in `DOCS_PROGRESS.md`.

## Proposed Information Architecture

The future browser documentation should use the following page tree.

```text
docs/
  index.md
  getting-started/
    installation.md
    quickstart.md
    task-selection.md
  concepts/
    evaluation-driven-development.md
    task-types.md
    feature-rules.md
    run-config.md
    alignment.md
    metrics.md
    logging.md
    visualization.md
  how-to/
    evaluate-single-feature.md
    evaluate-single-entity.md
    evaluate-multi-entity.md
    configure-feature-rules.md
    interpret-results.md
    save-logs.md
    generate-visualizations.md
  reference/
    overview.md
    config.md
    orchestrator.md
    logger.md
    visualization.md
    models.md
  contributor/
    docs-style-guide.md
    docs-maintenance.md
    ai-handoff.md
  planning/
    DOCS_ARCHITECTURE_PLAN.md
    DOCS_PROGRESS.md
```

Notes:

- `planning/` is for repository contributors and AI agents, not end users.
- `reference/` is intended to be mostly generated from docstrings or closely tied to them.
- `how-to/` pages should be cookbook-style and task-driven.
- `concepts/` pages should explain semantics and design decisions.

## Proposed Navigation Model

The future MkDocs navigation should look like this:

```text
Home
Getting Started
  Installation
  Quickstart
  Choosing a Task Type
Concepts
  Why This Library Exists
  Task Types
  Feature Rules
  Run Config
  Alignment
  Metrics
  Logging
  Visualization
How-To Guides
  Evaluate a Single-Feature Task
  Evaluate a Single-Entity Task
  Evaluate a Multi-Entity Task
  Configure Feature Rules
  Interpret Results
  Save Logs
  Generate Visualizations
API Reference
  Overview
  config
  orchestrator
  logger
  visualization
  models
Contributor Docs
  Documentation Style Guide
  Documentation Maintenance
  AI Handoff
Planning
  Docs Architecture Plan
  Docs Progress
```

## Content Requirements Per Page

### `index.md`

Must include:

- one-paragraph library summary
- supported task types
- quick links to quickstart, task selection, and API reference
- one small architecture diagram or numbered flow

### `getting-started/installation.md`

Must include:

- `uv`-first install instructions
- development install instructions
- how to run tests
- optional extras for examples / visualization

### `getting-started/quickstart.md`

Must include:

- shortest realistic example
- link to task-type selection page
- link to `FeatureRule` and `RunConfig` pages

### `getting-started/task-selection.md`

Must include:

- decision table for `SINGLE_FEATURE`, `SINGLE_ENTITY`, `MULTI_ENTITY`
- input/output shape examples
- common misclassification mistakes

### `concepts/task-types.md`

Must include:

- exact meaning of each task type
- what alignment strategy each uses
- whether `index_key_name` is required
- whether entity presence metrics apply

### `concepts/evaluation-driven-development.md`

Must include:

- the practical motivation for the library
- a concise explanation of evaluation-driven development
- the role of domain experts and gold-set design
- what work the library standardizes vs what it does not replace
- example mappings from real tasks to `SINGLE_FEATURE`, `SINGLE_ENTITY`, and `MULTI_ENTITY`

### `concepts/feature-rules.md`

Must include:

- full explanation of every `FeatureRule` field
- allowed `feature_type` values
- defaults
- text, number, date, and category semantics
- examples for alias maps and tolerances

### `concepts/run-config.md`

Must include:

- every `RunConfig` field
- task-type-specific requirements
- interaction between `RunConfig`, `MatchingConfig`, and `ClassificationConfig`
- common invalid configurations

### `concepts/alignment.md`

Must include:

- `IndexAligner` vs `EntityAligner`
- weighted matching behavior
- threshold behavior
- tie-breaking and determinism

### `concepts/metrics.md`

Must include:

- macro precision / recall / F1 / specificity
- micro accuracy
- row accuracy
- entity presence metrics
- how optional values affect metrics

### `concepts/logging.md`

Must include:

- what `RunLogger` writes
- output file naming and location behavior
- how to interpret a log file

### `concepts/visualization.md`

Must include:

- what data the plotting functions expect
- which plots are task-specific
- behavior on empty inputs

### `how-to/*`

Each how-to page must include:

- a concrete scenario
- complete runnable example
- expected output or result interpretation
- links back to relevant concept pages

### `reference/*`

Each reference page must document:

- signature
- parameter names
- allowed values
- defaults
- return types
- side effects
- exceptions or error conditions

Minimum public API to cover:

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
- visualization plotting helpers

## Authoritative Source Files

When writing docs, use these files as the implementation source of truth:

- `src/extraction_testing/config.py`
- `src/extraction_testing/orchestrator.py`
- `src/extraction_testing/tests.py`
- `src/extraction_testing/aligners.py`
- `src/extraction_testing/utils.py`
- `src/extraction_testing/metrics.py`
- `src/extraction_testing/logger.py`
- `src/extraction_testing/models.py`
- `src/extraction_testing/visualization.py`
- `examples/test.py`
- `tests/test_visualization.py`
- `tests/test_task_types.py`

If documentation and code disagree, the code wins.

## AI-Agent Work Breakdown

The documentation work should be split into independent packages so multiple agents can work in parallel.

### Work Package 1: Site scaffolding

Scope:

- add MkDocs config
- add docs theme/tooling setup
- create empty page files matching the page tree

Dependencies:

- none

### Work Package 2: Getting Started

Scope:

- installation
- quickstart
- task selection

Dependencies:

- page tree must be approved

### Work Package 3: Concepts

Scope:

- task types
- feature rules
- run config
- alignment
- metrics
- logging
- visualization

Dependencies:

- page tree must be approved

### Work Package 4: API reference

Scope:

- reference pages
- docstring improvements if needed
- generated API pages or hand-authored stubs

Dependencies:

- decision on generated vs hand-authored reference format

### Work Package 5: How-to guides

Scope:

- one page per common workflow

Dependencies:

- concept pages should exist first

### Work Package 6: Contributor docs

Scope:

- docs style guide
- docs maintenance guide
- AI handoff guide

Dependencies:

- planning docs already exist

## Documentation Style Requirements

All docs contributors should follow these rules:

- Use the new task-type vocabulary consistently: `SINGLE_FEATURE`, `SINGLE_ENTITY`, `MULTI_ENTITY`.
- Prefer explicit tables for parameter documentation.
- Always document defaults and task-specific requirements.
- Show `uv` commands first.
- Keep examples minimal but runnable.
- Do not describe behavior that is not implemented.
- Link concept pages to reference pages, and vice versa.

## AI-Agent Handoff Protocol

Every AI agent working on docs should:

1. Read this file.
2. Read `DOCS_PROGRESS.md`.
3. Claim a work package or page set in `DOCS_PROGRESS.md`.
4. Complete only the claimed scope.
5. Update `DOCS_PROGRESS.md` before finishing.

Each handoff update should include:

- date
- agent identifier or label
- pages touched
- status change
- unresolved questions
- follow-up suggestions

## Open Decisions

These decisions should be resolved before full docs implementation:

- whether API reference should be fully generated with `mkdocstrings` or a hybrid of generated and hand-written pages
- whether to include diagrams in concept pages
- whether to publish docs through GitHub Pages only or another hosted docs platform
- whether to add an AI-oriented summary artifact such as `llms.txt` or `docs/contributor/ai-handoff.md`

Recommended answers:

- use a hybrid approach: hand-written conceptual pages plus generated API reference
- include simple diagrams only where they clarify alignment or evaluation flow
- publish through GitHub Pages first
- add an AI-oriented handoff page in the docs, but do not build an MCP first

## Definition of Done For Documentation v1

The first documentation release is complete when:

- all pages in the proposed tree exist
- getting-started pages are written
- core concept pages are written
- reference pages cover the public API
- examples use current task-type names
- local browser rendering works
- docs are navigable without reading the README first
- `DOCS_PROGRESS.md` shows no critical blockers
