# DWMRI implementation planning — reference

## Decision tree

Walk branches in order. Skip with "N/A — …" when clearly irrelevant.

### 1. Goal and scope

- What is being built? (new model, extend existing pipeline, baseline runner, eval harness, experiment manifest, utility)
- What is explicitly **out of scope** for this iteration?
- Success criteria: what must work before the change is considered done?

### 2. Package placement

Existing first-party packages (prefer extending over new packages):

| Package | Typical use |
| --- | --- |
| `drcnet_hybrid_rgs` | DRCNet hybrid RGS training/reconstruction |
| `restormer_hybrid_rgs` | Restormer hybrid RGS (parallel structure to DRCNet) |
| `mds2s` | Multi-dimensional signal-to-signal / self-supervised |
| `p2s` | Patch2Self-related tooling |
| `paper_eval` | Paper-style evaluation, DTI metrics, baseline runners |
| `utils` | Shared config, checkpoints, metrics, reproducibility |

Decision: extend which package, or create a new top-level package under `src/`?

### 3. Module pattern

Standard pipeline layout:

- `config.yaml` — profile blocks (`dbrain`, `stanford`) and training/data/model sections
- `run.py` — CLI / orchestration
- `data.py`, `fit.py`, `model.py` — loading, training loop, architecture

Decision: which modules are new vs modified? Which YAML profile(s) apply?

### 4. Data and configs

- Which dataset / profile (`dbrain`, `stanford`, other)?
- D-Brain overrides: use `dbrain.` prefix for profile fields (see [experiments/AGENTS.md](../../../experiments/AGENTS.md))
- Progressive training: if `train.progressive.enabled` is true, `train.num_epochs` may not cap total training — disable for short pilots

### 5. Experiments integration

- Does this need a new manifest job, driver change, or sweep generator?
- Registry / baseline registration in [BASELINES_REGISTRY.md](../../../experiments/BASELINES_REGISTRY.md)?
- Pilot vs production params — see [PILOT_vs_PRODUCTION.md](../../../experiments/PILOT_vs_PRODUCTION.md) when relevant

### 6. Training paradigm

- Supervised vs self-supervised / blind-spot / hybrid (Scheme 2)
- If self-supervised or hybrid: apply [j-invariance-dwmri-reviewer](../j-invariance-dwmri-reviewer/SKILL.md) — J-set, mask convention, assumptions, threats
- Transfer learning, ensembling, or one-model-per-volume patterns if applicable

### 7. Evaluation and metrics

- Metrics via `paper_eval`? Which baselines to compare against?
- Inference-only path (`--skip-train`, checkpoint handling)?

### 8. Reproducibility

- `train.seed`, `train.reproducible` in config
- Shared helpers in `utils/repro_seed.py`

### 9. Tech debt and risks

- Any unvalidated assumptions → mark with `# TODO: tech debt — …` at implementation time
- Known threats: memory for 3D volumes, GPU determinism, correlated DWI noise, motion

### 10. Validation

- `ruff check src` / `ruff format src` after Python changes
- `pytest` for new or touched test coverage
- Smoke command or pilot manifest job for end-to-end check

---

## Plan output template

Use this structure for the final deliverable:

```markdown
# [Implementation title]

## Summary
[One paragraph: what, why, and expected outcome]

## Decisions (resolved)
| Topic | Decision | Rationale |
| --- | --- | --- |
| … | … | … |

## Open questions / deferred
- [Question or branch] — deferred because …

## Files and packages to touch
- `path/to/file` — [create | modify] — brief note

## Implementation steps (ordered)
1. …
2. …

## Validation / test plan
- [ ] …

## Risks and tech-debt notes
- …

## Suggested skills for execution
- [e.g. j-invariance-dwmri-reviewer if hybrid/self-supervised]
- [e.g. dl-expert rule for architecture choices]
```
