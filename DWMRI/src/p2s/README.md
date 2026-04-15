# P2S — Patch2Self Denoising Pipeline

Patch2Self baseline for full DWI reconstruction, with two selectable backends:

| Backend | Description |
|---------|-------------|
| `dipy` (default) | [DIPY Patch2Self](https://docs.dipy.org/stable/examples_built/preprocessing/denoise_patch2self.html) — global sklearn OLS/ridge regressor with intensity shift |
| `sklearn_reference` | Volume hold-out regression adapted from [Kang et al. MD-S2S](https://github.com/B9Kang/MD-S2S-Multidimensional-Self2Self/blob/main/model_patch2self.py) — for each DWI volume, all other volumes serve as predictors; implemented in `p2s/sklearn_patch2self.py` |

Both backends leave b0 volumes (bval ≤ `b0_threshold`) unchanged and denoise only diffusion-weighted volumes.

## Weights & Biases

Runs log to project `DWMRI-Denoising` by default (same as MDS2S), with tags `Patch2Self`, `P2S`, and the dataset name. Logged fields include data shape/stats, denoising wall time, denoised min/max/mean, output NIfTI path, reconstruction metrics (dBrain), and comparison images.

Disable W&B for offline runs:

```yaml
wandb:
  enabled: false
```

Or in code: `main(dataset="stanford", use_wandb=False)`.

## Configuration

Edit `p2s/config.yaml` to adjust per-dataset settings.

### Backend selection

```yaml
patch2self:
  backend: "dipy"          # or "sklearn_reference"
```

### DIPY backend options

| Key | Description |
|-----|-------------|
| `patch2self.model` | `"ols"` (default, accurate) or `"ridge"` (faster) |
| `patch2self.shift_intensity` | Shift per-volume intensities before denoising |
| `patch2self.clip_negative_vals` | Clip negatives in output (prefer `shift_intensity`) |
| `patch2self.b0_threshold` | Volumes with `bval <= threshold` are treated as b0 |

### sklearn_reference backend options

| Key | Default | Description |
|-----|---------|-------------|
| `patch2self.sklearn_model` | `"ols"` | Regressor: `"ols"`, `"ridge"`, `"lasso"`, `"mlp"` |
| `patch2self.patch_radius` | `[0, 0, 0]` | Spatial half-width of patches. `[0,0,0]` = single-voxel (pure inter-volume regression, minimal RAM). `[1,1,1]` = 3×3×3 neighbourhood |
| `patch2self.patch_stride` | `1` | Spatial grid step. `1` = every voxel (dense). `2` = every other voxel. Higher values cut memory and compute |
| `patch2self.use_b0_as_predictors` | `true` | Include b0 volumes as fixed predictor channels for every DWI target |

#### Memory and compute considerations

For a dense single-voxel run (`patch_radius=[0,0,0]`, `stride=1`) with Stanford HARDI (81×106×76, 150 DWI):

- The patch matrix has shape `(n_predictor_vols, n_voxels, 1)` — one feature per predictor, one row per voxel.
- Each volume trains a fresh sklearn regressor with `n_samples = (n_pred_vols−1) × n_voxels` and `n_features = 1`.
- For `"ols"` with `n_jobs=-1` this is parallelised across cores but can still take several minutes per volume.
- To experiment quickly, use `patch_stride: 2` or `patch_stride: 4` to subsample the patch grid, or `sklearn_model: "ridge"` which is faster than OLS on large matrices.

### Common options

| Key | Description |
|-----|-------------|
| `data.noise_sigma` | Added Rician noise level (for dBrain synthetic noise) |
| `reconstruct.metrics_roi_threshold` | Voxel threshold for tissue-only ROI metrics |

## Running

### Stanford HARDI (built-in DIPY dataset)

```bash
cd src
python runner_stanford_p2s.py
```

Or programmatically:

```python
from p2s.run import main
main(dataset="stanford", reconstruct=True, generate_images=True)
```

### dBrain

```bash
cd src
python -m p2s.run          # defaults to dbrain
```

Or programmatically:

```python
from p2s.run import main
main(dataset="dbrain", reconstruct=True, generate_images=True)
```

## Outputs

| Path | Description |
|------|-------------|
| `p2s/output/{dataset}/bvalue_{b}/noise_sigma_{s}/denoised_patch2self.nii.gz` | Denoised 4D NIfTI |
| `p2s/metrics/{dataset}/bvalue_{b}/noise_sigma_{s}/metrics.json` | Full-image MSE/PSNR/SSIM (dBrain only) |
| `p2s/metrics/{dataset}/bvalue_{b}/noise_sigma_{s}/metrics_roi.json` | Tissue-only ROI metrics (dBrain only) |
| `p2s/images/{dataset}/bvalue_{b}/noise_sigma_{s}/` | Visual comparison PNG files |

**Stanford**: no ground truth is available so `metrics.json` records `{"note": "no_reference_available"}` and images show noisy vs denoised side by side.

## Requirements

Requires `scikit-learn>=1.0.0` (used by both DIPY Patch2Self internally and the sklearn_reference backend).
