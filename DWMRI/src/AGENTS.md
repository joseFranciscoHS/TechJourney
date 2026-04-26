# Agent guide: `src/` (Python package)

Packages are discovered from `src/` via `pyproject.toml` (`tool.setuptools.packages.find`). Treat each top-level directory below as a **first-party** import root (Ruff `known-first-party` matches these names).

## Packages (high level)

| Package | Role |
| --- | --- |
| `mds2s` | Multi-dimensional signal-to-signal / self-supervised denoising path; CLI entry `mds2s` → `mds2s.run:main`. |
| `p2s` | Point-to-signal style processing (Patch2Self-related tooling, configs). |
| `drcnet_hybrid_rgs` | DRCNet hybrid RGS pipeline: 3D/2D entrypoints (`run.py`, `run2d.py`, `run2d_hybrid.py`, Stanford few-volume scripts). |
| `restormer_hybrid_rgs` | Restormer hybrid RGS pipeline; parallel structure to DRCNet package. |
| `paper_eval` | Paper-style evaluation helpers (e.g. DTI metrics, baseline runners). |
| `utils` | Shared helpers: config loading, checkpoints, metrics, image utilities, `repro_seed`, etc. |

## Typical module pattern

Many pipelines follow:

- `config.yaml` — YAML profile blocks (e.g. `dbrain`, `stanford`) and training/data/model sections.
- `run.py` — CLI / orchestration (training, reconstruction, registry paths).
- `data.py` / `fit.py` / `model.py` — loading, training loop, architecture.

Prefer extending existing patterns over introducing one-off structures.

## Reproducibility

- `train.seed` in package `config.yaml` files seeds Python, NumPy, and PyTorch where wired through the pipeline.
- `train.reproducible: true` tightens determinism (slower; GPU runs may still differ bitwise).
- Shared helpers live in `utils/repro_seed.py` (e.g. cuDNN behavior).

See [README.md](README.md) for a longer reproducibility note.

## Imports

First-party names: `drcnet_hybrid_rgs`, `mds2s`, `p2s`, `restormer_hybrid_rgs`, `paper_eval`, `utils` — keep imports consistent with Ruff isort config in the repo root `pyproject.toml`.

## Upstream context

Parent agent entry: [../AGENTS.md](../AGENTS.md).
