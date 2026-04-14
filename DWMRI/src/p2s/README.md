# P2S — Patch2Self Denoising Pipeline

Classical [DIPY Patch2Self](https://docs.dipy.org/1.12.0/examples_built/preprocessing/denoise_patch2self.html) baseline for full DWI reconstruction.

Denoises **only diffusion-weighted volumes** (`bvals > b0_threshold`); b0 volumes are kept intact (`b0_denoising=False`).

## Configuration

Edit `p2s/config.yaml` to adjust per-dataset settings:

| Key | Description |
|-----|-------------|
| `patch2self.model` | `"ols"` (default, accurate) or `"ridge"` (faster) |
| `patch2self.shift_intensity` | Shift per-volume intensities before denoising (recommended `true`) |
| `patch2self.clip_negative_vals` | Clip negatives in output; use `shift_intensity` instead |
| `patch2self.b0_threshold` | Volumes with `bval <= threshold` are treated as b0 |
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
python runner_dbrain_p2s.py
```

Or programmatically:

```python
from p2s.run import main
main(dataset="dbrain", reconstruct=True, generate_images=True)
```

Or via CLI default:

```bash
cd src
python -m p2s.run
```

## Outputs

| Path | Description |
|------|-------------|
| `p2s/output/{dataset}/bvalue_{b}/noise_sigma_{s}/denoised_patch2self.nii.gz` | Denoised 4D NIfTI |
| `p2s/metrics/{dataset}/bvalue_{b}/noise_sigma_{s}/metrics.json` | Full-image MSE/PSNR/SSIM vs reference (dBrain only) |
| `p2s/metrics/{dataset}/bvalue_{b}/noise_sigma_{s}/metrics_roi.json` | Tissue-only ROI metrics (dBrain only) |
| `p2s/images/{dataset}/bvalue_{b}/noise_sigma_{s}/` | Visual comparison PNG files |

**Stanford**: no ground truth is available so `metrics.json` records `{"note": "no_reference_available"}` and images show noisy vs denoised.

## Requirements

Requires `scikit-learn>=1.0.0` in addition to the base repo dependencies (used internally by DIPY Patch2Self v3).
