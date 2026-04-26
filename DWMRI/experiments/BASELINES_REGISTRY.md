# Baselines (MP-PCA) and hybrid `registry.jsonl` (v1)

This document relates outputs from the classical baseline script **`/Users/francisco/Documents/TechJourney/TechJourney/DWMRI/src/paper_eval/baselines/mppca_run.py`** to the **hybrid** run registry produced by `drcnet_hybrid_rgs.run` and `restormer_hybrid_rgs.run` (`schema_version: "v1"`, one JSON object per line).

## What `mppca_run.py` writes

The script (DIPY `mppca`):

- Reads **`--noisy`** and **`--gt`** as **NumPy** arrays, shape **(X, Y, Z, V)**.
- Writes under **`--out-dir`**:
  - `denoised.npy` — denoised DWI array.
  - `sigma.npy` — per-voxel (or as returned) noise scale map from MP-PCA.
  - `metrics.json` / `metrics_roi.json` — image-space metrics (via `utils.metrics` / `save_metrics` in that package).
  - DTI file(s) from `save_dti_metrics` with **placeholder** nulls: `{"fa_mae": null, "md_mae": null, "ad_mae": null, "rd_mae": null}` (no DTI fit in the baseline path as written).

There is **no** JSONL row append, no WandB hook, and no `exp_id` / `job_id` in the script.

## What hybrid `run.py` appends to `registry.jsonl` (v1)

Each run appends a single JSON line that includes, among other fields (non-exhaustive):

| Field (concept) | Meaning |
| --- | --- |
| `schema_version` | Currently `"v1"`. |
| `exp_id`, `job_id` | Experiment and job identifier from CLI (`--exp-id`, `--job-id`). |
| `recipe` | From `--recipe` (optional; manifest often sets this). |
| `status` | `success` or `failed`. |
| `error` | String if failed, else `null` / empty. |
| `timestamps` | `start_utc`, `end_utc` (ISO). |
| `duration_s` | Wall time. |
| `stage` | e.g. `train_reconstruct`, `train`, or `reconstruct`. |
| `dataset`, `regime` | e.g. `dbrain`, `self_supervised`. |
| `architecture` | e.g. `drcnet` or `restormer`. |
| `dimensionality` | e.g. `3d` (for `run.py`). |
| `sampling_mode` | From `data.shell_sampling_mode` (e.g. `rgs` / `sequential`). |
| `sampling_config` | `g_shell`, `k_input`, `target_channel`, `window_policy`. |
| `inference_config` | `n_context_samples`, `n_preds` from `reconstruct`. |
| `train_config` | `epochs`, `batch_size`, `lr`, `progressive_enabled`. |
| `control_metrics` | `n_params`, timing, `peak_gpu_mem_mb`, etc. |
| `quality_metrics_full`, `quality_metrics_roi` | Structured metric dicts. |
| `dti_metrics` | Populated when DTI is computed; contrast with MP-PCA placeholder. |
| `hardware` | Host / GPU info. |

## Relating MP-PCA results to a registry row (manual or future wrapper)

1. **Folder layout**: For apples-to-apples comparison, it helps to use the same `out_dir` naming as hybrid metrics (e.g. a sibling directory under a shared `output_root`), even though MP-PCA does not use `output_root` remapping in the script.
2. **NIfTI vs `.npy` gap**: The hybrid training stack uses NIfTI paths in `config.yaml` and internal loaders; the baseline expects **pre-exported** `.npy` volumes. A fair comparison still requires the **same** preprocessing (clipping, scaling, ROI definition) and the **same** metric definitions as `compute_metrics` in the training codebase.
3. **Manual JSONL row (template)**: For a one-off “baseline as a family member” in the same JSONL file, you can append a hand-written v1 line that mirrors hybrid keys, reusing the same `exp_id` and a new `job_id` (e.g. `mppca_dbrain_001`), with `architecture: "mp_pca"`, `stage: "reconstruct"`, and `quality_metrics_full` / `quality_metrics_roi` **copied or transcribed** from the baseline `metrics.json` files. Keep `schema_version: "v1"`.
4. **Future wrapper**: A small script could: run `mppca_run.py`, read `metrics.json` / `metrics_roi.json`, compute real DTI MAEs if you add DIPY to that path, and emit a v1 line compatible with the hybrid schema.

## Summary

- **MP-PCA** = folder of arrays + `metrics*.json` + DTI nulls.  
- **Hybrid runs** = full training + reconstruction + `registry.jsonl` v1 with rich provenance.  
- Bridging them is a **convention and tooling** problem (especially **`.npy` export** and **unified DTI**), not a schema lock-in: v1 is wide enough to represent baselines with a few consistent field choices and a non-neural `architecture` label.
