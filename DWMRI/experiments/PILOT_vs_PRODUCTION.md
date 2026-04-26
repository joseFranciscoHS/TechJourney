# Pilot vs production (hybrid RGS 3D)

This note compares the **pilot** jobs in `manifest_pilot.yaml` to the **production-style** defaults in `drcnet_hybrid_rgs/config.yaml` and `restormer_hybrid_rgs/config.yaml` for the `dbrain` profile. 2D entrypoints use `drcnet_hybrid_rgs.run2d` / `restormer_hybrid_rgs.run2d` and are in the manifest only as quick shakedowns, not as full paper ablations.

## Configuration table (D-Brain)

| Key / concept | Pilot (3D, manifest) | Production default (`dbrain` in package `config.yaml`) |
| --- | --- | --- |
| `train.progressive.enabled` / `stages` | `false` (3D jobs); not applicable when disabled | `true` with **3 stages** (patch size / `epochs` per stage differ by package: DRCNet uses 100 epochs per stage; Restormer uses 30 per stage; see the YAML) |
| `train.num_epochs` | `5` (intended; requires `dbrain.train.*` overrides—see below) | DRCNet: `20` — **overridden in practice** when progressive training is on (see warning). Restormer: `30` — same caveat. |
| `train.supervised` | Inherits from config / `--regime` (default self-supervised, masked MSE) | DRCNet `dbrain`: `false`. Restormer `dbrain`: key omitted (self-supervised path as in `run.py`). For explicit flag see `stanford` block in Restormer config. |
| `data.shell_sampling_mode` | `sequential` (pilot) | `rgs` (random K-subset) |
| `data.num_input_volumes` (K) / `model.in_channel` | `24` / `24` (unchanged) | `24` / `24` |
| `data.target_channel` | `23` | `23` (0-based; K-th slot) |
| `data.shell_gradient_volumes` (G) | `60` | `60` (used when `shell_sampling_mode=rgs`) |
| `train.mask_p` | Inherits (typically `0.3`) | `0.3` |
| `reconstruct.n_context_samples` / `n_preds` | Inherits (DRCNet: 24 / 12; Restormer: 24 / 5) | Same defaults per package (see `reconstruct` block) |
| `--output-root` / `--registry-path` | Injected by `experiments/driver.py` (see that script) | `run.py` remaps `train.checkpoint_dir`, `reconstruct.metrics_dir`, `reconstruct.images_dir` under `output_root`; `registry_path` is a **JSONL** append of one record per run |

### Critical warning: progressive training and `num_epochs`

When **`train.progressive.enabled` is `true`**, the training loop is driven by **per-stage** `epochs` in `train.progressive.stages`. The top-level `train.num_epochs` value in the config (and any `train.num_epochs=…` you might log in a manifest) **does not cap** total training time in the progressive schedule. For short pilot runs, **disable progressive training** and then `train.num_epochs` is honored (see 3D jobs in `manifest_pilot.yaml`).

### CLI overrides and `dbrain`

`--set` applies to the **full** config object **before** the `dbrain` block is selected. To change D-Brain fields, use the **`dbrain.`** prefix, e.g. `dbrain.train.num_epochs=5`, `dbrain.data.shell_sampling_mode=sequential`, `dbrain.train.progressive.enabled=false`. Unqualified `train.*` / `data.*` would create or alter top-level keys and **not** the `dbrain` profile.

## WandB (pilot / cluster-friendly)

- Pass **`--no-wandb`** to `drcnet_hybrid_rgs.run` and `restormer_hybrid_rgs.run` when you do not want WandB to initialize (e.g. batch jobs without API keys).
- Optionally set the environment to offline logging: `WANDB_MODE=offline` (WandB still writes local artifacts if enabled; with `--no-wandb` the run does not initialize WandB at all—use whichever matches your needs).

## Inference-only (no training)

Point `--checkpoint` at a saved **`best_loss_checkpoint.pth`** and pass **`--skip-train`** (still runs reconstruction and registry append unless you also use `--skip-reconstruct` as needed for your job).

## DTI maps

With **`reconstruct.compute_dti: true`** (default in `dbrain` for both models), the pipeline can compute DTI-derived scalars and compare to GT (requires DIPY and matching b-vectors; see `config.yaml`).

## Barrido K (§1.2)

Use [`generate_k_sweep_jobs.py`](generate_k_sweep_jobs.py) to emit a manifest with coherent `dbrain.model.in_channel`, `dbrain.data.num_input_volumes`, `dbrain.data.target_channel=K-1`, and `dbrain.data.shell_gradient_volumes=G` for each K. Overrides use the **`dbrain.`** prefix so they apply after dataset selection. When driving jobs with [`driver.py`](driver.py), its trailing `--output-root` overrides any `--output-root` embedded in the manifest command; checkpoint paths still differ by K via `sequential_G…_K…` segments when `output_root` is shared.

## Run checklist (`exp_id` / `job_id`)

When using the driver or ad hoc CLI:

- [ ] Set a unique **`--exp-id`** for the whole experiment (or campaign).
- [ ] Set a stable **`--job-id`** per manifest job (or per CLI invocation) so the registry line is unique and traceable.
- [ ] Confirm **`--registry-path`** points to the JSONL you intend to append to.
- [ ] Confirm **`--output-root`** if you are isolating checkpoints and metrics from the in-repo default paths.

## “Hueco 2D” / plan §1.4 (paper `plan_para_escribir_el_paper.md`)

**`run2d`** in this repo is a **pilot / engineering shakedown** only. It is **not** paper–parity for the ablation in **§1.4 (3D vs Conv2D slice-by-slice)** until the 2D path is fully aligned (same RGS+masking protocol, reporting, and registry conventions as 3D). Do not treat 2D pilot metrics as the published 2D-vs-3D ablation.
