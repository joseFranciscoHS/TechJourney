# DRCNet Hybrid – One Model per Volume

Train one **independent** `DenoiserNet` per DWI volume. No transfer learning: each model learns to denoise a single target volume only, following **J-invariance** (spatial–angular hybrid MD-S2S). For N volumes you get N trained models.

## Method (J-invariance)

This pipeline implements **Scheme 2** from the J-invariance DWMRI denoising report: **MD-S2S style (spatial–angular hybrid)**.

- **Input:** All volumes \(v_{-j}\) plus the **target** volume \(v_j\) under a **pixel-level Bernoulli mask**.
- **Loss:** Masked MSE computed **only over the masked (occluded) pixels** of the target volume.
- **Rationale:** The network can use both other gradient volumes and the spatial context of visible pixels in the target volume, while J-invariance (no access to the noisy values of occluded pixels) ensures training without clean ground truth.

See `src/J_invariance_DWMRI_denoising_report.md` for the full theoretical background.

## Architecture

- **Model:** `DenoiserNet` — 3D CNN with gated blocks, factorized convolutions, and residual connections. Input channels = number of volumes (all volumes, including the masked target); output channels = 1 (predicted target volume).
- **Training data:** One shared `TrainingDataSet` is built once from sliding windows over the 4D DWMRI. Before each volume’s training run, `set_source_volume_index(i)` is called so that every sample uses volume `i` as the target. The target volume is Bernoulli-masked in the input; the model predicts the target; loss is masked MSE on the target.
- **Patch filtering:** Background-only patches can be excluded from training to reduce positive bias in empty regions (see [Patch Filtering](#patch-filtering) below).
- **Per-volume training:** For each volume index `i` in `0 .. num_volumes-1`, a **new** `DenoiserNet` is created, trained on that volume only, and saved under `checkpoint_dir/.../volume_{i}/` (e.g. `best_loss_checkpoint.pth`, `latest_checkpoint.pth`).

## Patch Filtering

To reduce noise in background (empty) regions, the training pipeline can exclude patches that contain little or no signal. This aligns with the original DRCnet implementation which only trains on patches containing tissue.

**Why?** With Rician noise, empty regions have a small positive expected value (Rayleigh distribution). MSE loss encourages the model to predict this positive value rather than zero. By excluding background patches, the model focuses on tissue denoising.

**Configuration** (`config.yaml` under `data`):

| Parameter | Description |
|-----------|-------------|
| `patch_filter_method` | `"none"` (keep all), `"threshold"` (signal-based), or `"otsu"` (brain mask) |
| `min_signal_threshold` | For `"threshold"`: exclude patches where `max(clean_patch) <= threshold` (default: 0.02) |
| `otsu_median_radius` | For `"otsu"`: median filter radius for DIPY `median_otsu` (default: 2) |
| `otsu_numpass` | For `"otsu"`: number of median filter passes (default: 1) |

**Example:**
```yaml
data:
  patch_filter_method: "threshold"
  min_signal_threshold: 0.02
```

## Reconstruction

- One model per volume: for each volume index, the corresponding checkpoint is loaded (`volume_{i}/best_loss_checkpoint.pth`), then `reconstruct_single_volume` is run with the same Bernoulli-masking strategy and multiple predictions averaged (`n_preds`). Results are concatenated into the full DWI and used for metrics and visualizations.

## How to run

From the project root (or with `src` on `PYTHONPATH`):

```python
from src.drcnet_hybrid_multiple_networks.run import main

# Train one model per volume, then reconstruct
main(dataset="dbrain", train=True, reconstruct=True, generate_images=True)
```

Or run the module:

```bash
cd src && python -m drcnet_hybrid_multiple_networks.run
```

**Arguments:** `dataset` (`"dbrain"` | `"stanford"`), `train`, `reconstruct`, `generate_images`, and `transfer_learn` (accepted but **ignored** in this pipeline). Default: `main(dataset="dbrain")` with `train=True`, `reconstruct=True`, `generate_images=True`.

## Config

- **config.yaml** provides per-dataset sections: `dbrain` and `stanford`, each with `train`, `reconstruct`, `model`, and `data`. No separate `transfer` section.
- **Checkpoints** are stored under paths that include dataset, b-value, number of volumes, noise sigma, and learning rate, then one subdir per volume:
  - e.g. `drcnet_hybrid_multiple_networks/checkpoints/dbrain/bvalue_2500/num_volumes_10/noise_sigma_0.1/learning_rate_0.0015/volume_0/`, `volume_1/`, …
- **Losses** (per-epoch logs): `drcnet_hybrid_multiple_networks/losses/<dataset>/bvalue_.../num_volumes_.../noise_sigma_.../learning_rate_.../volume_{i}/`.
- **Reconstruction** expects `checkpoint_dir/volume_{i}/best_loss_checkpoint.pth` for each volume `i`. Metrics and images are written under `reconstruct.metrics_dir` and `reconstruct.images_dir` with the same parameter-based subdir structure.
- **Training:** Mask probability `mask_p`, scheduler (e.g. `reduceLROnPlateau`), multi-GPU options, and data (patch size, step, num_volumes, paths, patch filtering) are in the config. **Reconstruction:** `mask_p` and `n_preds` (number of masked predictions to average per volume) are under `reconstruct`.
- **Patch filtering:** `patch_filter_method`, `min_signal_threshold`, `otsu_median_radius`, `otsu_numpass` under `data` (see [Patch Filtering](#patch-filtering)).

## Files

| File | Role |
|------|------|
| `run.py` | Entry point: load config and data, train one model per volume, then reconstruct and (optionally) compute metrics and save comparison images; supports wandb. |
| `model.py` | `DenoiserNet`, `GatedBlock`, `DenoisingBlock`, `FactorizedBlock`. |
| `data.py` | `TrainingDataSet`, sliding windows over 4D data, Bernoulli masking of target volume, `set_source_volume_index`, patch filtering (threshold/otsu). |
| `fit.py` | `fit_model`: masked MSE loss, checkpointing, optional loss tracker and wandb; `fit_transfer_wrapper` for transfer (not used in this pipeline). |
| `reconstruction.py` | `reconstruct_single_volume` (one volume, one model), `reconstruct_dwis` (all volumes with one model per volume); masked input + `n_preds` averaging. |
| `config.yaml` | `dbrain` / `stanford` train, reconstruct, model, data. |
