# Agent guide: `experiments/`

Batch and pilot workflows for hybrid RGS runs. Manifests list shell commands; `driver.py` executes them sequentially and records logs.

## Driver

- **Script:** `driver.py` — loads a YAML manifest (`--manifest`), requires `--exp-id`, `--output-root`, `--registry-path`.
- **Per job:** runs `job["command"]` with optional `cwd` (`repo_root` → parent of `experiments/`, else defaults to `../src` when that directory exists).
- **Flags appended** when `append_registry_flags` is not `false`: `--exp-id`, `--job-id`, `--recipe`, `--output-root`, `--registry-path`.
- **Logs:** `{output_root}/runs/{job_id}.log`; summary JSON: `{output_root}/driver_summary.json`.
- **Failure:** use `--fail-fast` to stop on first non-zero exit.

## Manifests and helpers

- Example / pilot manifests: `manifest_pilot.yaml`, `paper_pilot.yaml`, and related YAML in this folder.
- Job matrix / sweep generators: `generate_k_sweep_jobs.py`, `generate_matrix_jobs.py` (and similar) — read before changing manifest shape.
- Metrics merge / export utilities: e.g. `merge_pilot_metrics.py`, `paper_export_dbrain_volume_pair.py`.

When adding jobs, keep command lines consistent with how `drcnet_hybrid_rgs.run` / `restormer_hybrid_rgs.run` (and 2D entrypoints) expect CLI flags.

## D-Brain config overrides (`--set`)

Overrides apply to the **full** config **before** the `dbrain` profile is selected. To change D-Brain fields, use the **`dbrain.`** prefix, e.g. `dbrain.train.num_epochs=5`, `dbrain.data.shell_sampling_mode=sequential`, `dbrain.train.progressive.enabled=false`. Unqualified `train.*` / `data.*` can shadow or miss the profile block.

## Progressive training vs `num_epochs`

When `train.progressive.enabled` is `true`, training is driven by per-stage `epochs` in `train.progressive.stages`; top-level `train.num_epochs` may not cap total training. For short pilots, disable progressive training so `train.num_epochs` is honored. Details: [PILOT_vs_PRODUCTION.md](PILOT_vs_PRODUCTION.md).

## Useful runtime flags (training entrypoints)

- `--no-wandb` — avoid WandB init on clusters without API keys.
- `--skip-train` — skip training when pointing `--checkpoint` at e.g. `best_loss_checkpoint.pth` (still runs reconstruction/registry unless further skips are used).
- Inference-only / DTI notes: see [PILOT_vs_PRODUCTION.md](PILOT_vs_PRODUCTION.md).

## Baselines and registry

- Baseline inventory / conventions: [BASELINES_REGISTRY.md](BASELINES_REGISTRY.md).

## Upstream context

Parent agent entry: [../AGENTS.md](../AGENTS.md).
