# Documentation Style Guide

This guide defines how documentation should be written in this repository so it stays readable, easy to maintain, and useful for both humans and AI agents.

## Core principles

- Prefer task-first explanations over module-first explanations.
- Keep the code as the source of truth when examples or older docs drift.
- Be explicit about defaults, side effects, and failure modes.
- Keep pages concise, but not so short that important behavior becomes implicit.
- Use runnable examples when they make behavior easier to understand.

## Audience split

Write with two audiences in mind:

- end users who want to evaluate extraction outputs correctly
- maintainers and agents who need predictable structure and precise behavior notes

This usually means:

- concepts pages explain semantics
- how-to pages show workflows
- reference pages answer exact API questions
- contributor pages explain process and upkeep

## Page style

- Start with the most important answer first.
- Use short sections with informative headings.
- Prefer flat bullet lists over dense paragraphs when listing rules or outcomes.
- Avoid marketing language and avoid filler.
- Keep terminology consistent with the code: `SINGLE_FEATURE`, `SINGLE_ENTITY`, `MULTI_ENTITY`, `FeatureRule`, `RunConfig`, `ResultBundle`.

## Examples

Good documentation examples in this repo should be:

- complete enough to run or adapt
- minimal enough to scan quickly
- consistent with the package's declared Python support
- aligned with the actual public API

When writing examples:

- prefer small Pydantic models with only the fields needed for the point
- show the exact `RunConfig` and `FeatureRule` setup that matters
- include a short “what to expect” section when output values are not obvious

## Handling implementation caveats

Do not hide surprising behavior. If the code has a current limitation or caveat, document it clearly.

Examples in this repo include:

- fields that exist in config models but are not yet used in the runtime path
- differences between intended semantics and current implementation details
- visualization helpers that expect data shapes different from current evaluator outputs

When possible:

- describe the current behavior precisely
- avoid promising behavior the code does not implement
- leave future cleanup as a follow-up note rather than papering over it

## Linking and structure

- Link from getting-started pages into concepts and reference pages.
- Link from how-to pages back to the concepts they rely on.
- Keep the page tree aligned with `mkdocs.yml`.
- Add new pages intentionally; do not create ad hoc pages outside the planned structure unless the architecture plan is updated first.

## Reference-page expectations

Reference pages should answer:

- what can be imported?
- what are the constructor or function signatures?
- which arguments are required?
- what are the defaults?
- what is returned?
- what side effects or runtime errors should callers expect?

Use generated API blocks where useful, but keep concise hand-authored summaries above them so the page is still readable if docstrings are sparse.

## Maintenance discipline

- Update docs when behavior changes, not later.
- Update examples when public names or semantics change.
- Update `DOCS_PROGRESS.md` for all documentation work.
- Treat unfinished content as explicit backlog, not hidden drift.

## Related pages

- [Documentation Maintenance](docs-maintenance.md)
- [AI Handoff](ai-handoff.md)
