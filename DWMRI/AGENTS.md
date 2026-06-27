# Agent guide: DWMRI

This folder is the **Diffusion-weighted MRI** reconstruction and denoising workstream inside the TechJourney monorepo. Use this file as the entry point; drill down into `src/AGENTS.md` and `experiments/AGENTS.md` when touching those areas.

## Layout

- `src/` — installable Python package (`dwmri-processing` in `pyproject.toml`); models, training, evaluation utilities.
- `experiments/` — batch manifests, `driver.py`, pilot/production notes, job generators.
- `.cursor/rules/` — Cursor rules (e.g. deep-learning domain, J-invariance review).
- `.cursor/skills/j-invariance-dwmri-reviewer/` — skill for blind-spot / J-invariant denoising review.
- `.cursor/skills/dwmri-plan-me/` — user-invoked alias to start implementation planning.
- `.cursor/skills/dwmri-implementation-planning/` — grilling-style planning before new work.
- `.cursor/skills/dwmri-handoff/` — session handoff for agent continuity.

Human-oriented install and feature overview: [src/README.md](src/README.md).

## Setup and checks

From this directory (`DWMRI/`):

```bash
uv sync --extra dev
```

```bash
ruff check src
ruff format src
pytest
```

Dependencies are defined in `pyproject.toml` (pinned via `uv.lock` when present). Do not introduce ad-hoc install paths unless the task requires it.

## Conventions for agents

- **Scope:** Change only what the task needs. Avoid drive-by refactors and unrelated files.
- **Language:** Code, comments, and commit messages in **English** unless the user explicitly asks otherwise.
- **Tech debt:** If new or changed code relies on **unvalidated assumptions** or is **not production-ready**, add an inline comment on the relevant block: `# TODO: tech debt — …` (see `.cursor/rules/tech-debt-todos.mdc`). Do not litter trivial code with TODOs.
- **Docs:** Do not add new Markdown guides unless requested (these `AGENTS.md` files are the agreed exception).
- After substantive Python edits, run Ruff (and tests if available for the touched code).

## Where to read next

| Topic | File |
| --- | --- |
| Module map, configs, reproducibility | [src/AGENTS.md](src/AGENTS.md) |
| Manifests, driver, D-Brain overrides | [experiments/AGENTS.md](experiments/AGENTS.md) |
| Cursor domain rules | `.cursor/rules/` |
