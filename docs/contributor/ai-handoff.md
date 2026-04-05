# AI Handoff

This page is the operational guide for AI agents working on documentation in this repository.

## First steps for any docs task

Before editing docs:

1. read `AGENTS.md`
2. read `README.md`
3. read `docs/planning/DOCS_ARCHITECTURE_PLAN.md`
4. read `docs/planning/DOCS_PROGRESS.md`
5. inspect the relevant source files before assuming behavior

Do not rely only on older docs text when the code is easy to inspect directly.

## Claiming work

When starting a docs task:

- move the relevant work package in `DOCS_PROGRESS.md` to `in_progress`
- claim the specific page rows if the work is page-level
- keep ownership explicit so handoffs stay readable

Do not silently start work without updating the tracker first.

## While writing

- keep the current page tree and navigation model intact unless the architecture plan is intentionally changed
- write task-first, concise content
- document current behavior, including caveats
- prefer small runnable examples over abstract pseudo-code when examples help

If you discover implementation issues while documenting:

- document the current behavior accurately
- mention the caveat in `DOCS_PROGRESS.md` if it matters for future docs work
- do not silently “correct” documentation to match intended behavior if the code does something else

## Before finishing

Every docs agent should:

1. update `DOCS_PROGRESS.md`
2. move statuses forward only for actually completed work
3. append a new handoff entry
4. list blockers and follow-up recommendations explicitly
5. run a verification pass appropriate to the environment

## Verification expectations

Preferred checks:

- `uv run mkdocs build --strict`
- targeted tests or examples when docs depend on runtime behavior
- file-level checks for placeholder text, broken naming, or nav drift

If a full docs build is not possible in the current environment, say so in the handoff notes.

## Handoff note quality

A good handoff entry should say:

- what was completed
- which files changed
- which work package statuses changed
- any blockers or caveats
- the recommended next task

Keep entries short, factual, and easy for the next agent to act on.

## Related pages

- [Documentation Style Guide](docs-style-guide.md)
- [Documentation Maintenance](docs-maintenance.md)
