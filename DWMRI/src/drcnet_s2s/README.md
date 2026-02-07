# DRCNet-S2S: Spatial–Angular Hybrid DWMRI Denoising

DRCNet combined with an MD-S2S style (Multidimensional Self2Self) framework for self-supervised denoising of Diffusion-Weighted MRI (DWMRI). Implements **Scheme 2** from the J-invariance technical report: a hybrid approach that integrates angular redundancy (Patch2Self) with fine-grained spatial redundancy (Self2Self).

## Theoretical Background

### J-Invariance Framework

The module is grounded in the **J-invariance** framework for denoising without ground truth. A function \(f\) is J-invariant with respect to a partition \(J\) of the data dimensions if the prediction along dimensions in \(J\) does not depend on the noisy input values in those same dimensions.

Under the assumption that noise is statistically independent across dimensions while the true signal exhibits correlation, minimizing the error between prediction and noisy data is equivalent to minimizing the error with respect to the clean signal.

### MD-S2S Style (Spatial–Angular Hybrid)

This implementation applies J-invariance at the **pixel level** across the full 4D DWI:

- **Angular redundancy:** All gradient volumes are used as input. Underlying anatomical structures are consistent across acquisitions despite varying gradient directions.
- **Spatial redundancy:** A Bernoulli mask occludes a subset of pixels across all volumes. The network must predict occluded pixels using only visible pixels (the complement \(J^c\)).
- **Loss mechanism:** The loss is computed exclusively over the masked pixels of the target volume.
- **Variance reduction:** At inference, multiple predictions with different masks are averaged (dropout-based ensemble) to reduce variance.

### Data Layout

- **Input/Output:** `(Vols, X, Y, Z)` — volumes along the first axis, spatial dimensions follow.
- **Patches:** Extracted as `(N, Vols, X, Y, Z)` per patch.

## Module Structure

```
drcnet_s2s/
├── run.py           # Main entry: train, reconstruct, evaluate
├── data.py          # TrainingDataSet with Bernoulli masking
├── fit.py           # J-invariant masked MSE training loop
├── model.py         # DRCNet (gated, factorized 3D convolutions)
├── reconstruction.py  # Inference with mask averaging
├── config.yaml      # Dataset-specific configuration
└── README.md        # This file
```

## Usage

### Training

```bash
# From project root
python -m drcnet_s2s.run --dataset dbrain
# or
python src/drcnet_s2s/run.py
```

### Configuration

Key parameters in `config.yaml`:

| Parameter | Description | J-Invariance Role |
|-----------|-------------|-------------------|
| `mask_p` | Bernoulli mask probability (pixels with p > mask_p are visible) | Controls J-set size; see bias–variance tradeoff |
| `n_preds` | Number of inference passes to average | Variance reduction at test time |
| `in_channel` / `out_channel` | Number of volumes | Must match DWI gradient count |

### Python API

```python
from drcnet_s2s.data import TrainingDataSet
from drcnet_s2s.fit import fit_model
from drcnet_s2s.model import DenoiserNet
from drcnet_s2s.reconstruction import reconstruct_dwis
```

## Bias–Variance Tradeoff

The choice of `mask_p` affects the J-set size:

- **mask_p too low:** Fewer masked pixels → more data for prediction, but loss estimated from fewer samples → higher variance.
- **mask_p too high:** More masked pixels → less context (smaller \(J^c\)) → potential bias.

Recommended range: 0.2–0.4 (e.g., 0.3).

## References

See [J_invariance_DWMRI_denoising_report.md](../J_invariance_DWMRI_denoising_report.md) for the full theoretical framework and proposed training schemes.
