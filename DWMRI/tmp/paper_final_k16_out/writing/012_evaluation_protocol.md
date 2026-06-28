# 012 — Evaluation Protocol Specification

## Context for the LLM

You are completing the **Experiments** section (§3) of a research paper on self-supervised DW-MRI denoising. The TODO at line 128 requests detailed specifications for synthetic noise, metrics computation, and tensor fitting.

## Files to attach

| File | Purpose |
|------|---------|
| `paper/Sepulveda_dwmri_restormer.tex` | Current manuscript with TODO at line 128 |
| `src/utils/noise.py` | Rician noise implementation |
| `src/utils/metrics.py` | PSNR/SSIM computation |
| `src/paper_eval/dti_metrics.py` | DTI tensor fitting via DIPY |

## Prompt

Write a detailed evaluation protocol specification to replace the TODO at line 128. This content belongs at the end of §3.1 (Datasets and Protocols), after the paragraph introducing D-Brain and Stanford HARDI.

### Content to include

**1. Synthetic Rician Noise (D-Brain)**

The D-Brain experiments add synthetic Rician noise to the normalized clean phantom data. For each voxel with clean signal $s \in [0,1]$ and noise level $\sigma$, the noisy magnitude is:

$$
y = \sqrt{(s + \eta_1)^2 + \eta_2^2},
$$

where $\eta_1, \eta_2 \sim \mathcal{N}(0, \sigma)$ are independent Gaussian noise samples. The corrupted volume is clipped to $[0,1]$. This per-voxel independent model ensures that noise satisfies the J-invariance assumptions across masked spatial sites and across diffusion volumes. Experiments use $\sigma \in \{0.05, 0.10, 0.15, 0.20\}$ for the noise-robustness study, with $\sigma=0.10$ as the primary evaluation condition.

**2. Image-Domain Metrics**

**Full-image metrics**: Mean squared error (MSE), peak signal-to-noise ratio (PSNR), and structural similarity index (SSIM) are computed over the entire 4D array (X × Y × Z × V). PSNR uses the dynamic range $\max(\text{original})$:

$$
\text{PSNR} = 20 \log_{10} \left( \frac{\max(\text{original})}{\sqrt{\text{MSE}}} \right).
$$

SSIM is computed with `skimage.metrics.structural_similarity` using `data_range = max − min` and `channel_axis=3` to treat the 4D volume as a multichannel image.

**ROI metrics**: To avoid background bias, metrics are recomputed over a brain tissue region of interest (ROI). The ROI mask is defined as voxels where any diffusion volume exceeds intensity threshold 0.02 (equivalent to 2% of the normalized range). PSNR-ROI and MSE-ROI use only masked voxels; SSIM-ROI is computed on the bounding-box crop of the ROI to preserve local spatial structure for the SSIM algorithm.

**3. Diffusion Tensor Metrics**

Tensor-derived biomarkers (FA, MD, AD, RD) are computed via DIPY's `TensorModel` with ordinary least squares (OLS) fitting. Before tensor fitting, denoised DWI volumes are denormalized to the original phantom intensity scale using the stored per-volume min-max parameters. The gradient table is constructed from the acquisition b-values and b-vectors; b0 volumes (b ≤ 50 s/mm²) are prepended to the denoised DWI stack. Tensor maps are computed on both the denoised output and the clean ground truth (D-Brain) or reference acquisition (Stanford). For D-Brain, mean absolute error (MAE) is reported over the ROI for each tensor metric:

$$
\text{FA-MAE} = \frac{1}{|\Omega_{\text{ROI}}|} \sum_{i \in \Omega_{\text{ROI}}} |\text{FA}_{\text{denoised}, i} - \text{FA}_{\text{GT}, i}|.
$$

MD-MAE, AD-MAE, and RD-MAE are computed similarly. **Units**: FA is dimensionless (range [0,1]). MD, AD, and RD errors are reported in the units output by DIPY after denormalization; for D-Brain, these represent relative diffusivity magnitudes derived from the phantom simulation intensity scale and should be interpreted as comparative metrics across methods rather than as physical diffusivity values in mm²/s.

**4. Software and Settings**

- **Noise generation**: `src/utils/noise.py`, per-volume independent Rician model
- **Metrics**: `src/utils/metrics.py` (PSNR/SSIM), `src/paper_eval/dti_metrics.py` (DTI)
- **Tensor fitting**: DIPY 1.x `TensorModel` with default OLS, no regularization
- **ROI threshold**: 0.02 for both image metrics and DTI mask

## Expected output

A LaTeX section addition (300-400 words) that:
1. Specifies the Rician formula with proper math notation
2. Describes PSNR/SSIM computation including dynamic range and SSIM settings
3. Defines the ROI mask and explains ROI vs full-image metrics
4. Details the DTI fitting procedure with DIPY citation
5. Clarifies the diffusivity-error units and their interpretation
6. Lists software components with paths/citations

The text should be technical and precise, suitable for the Methods section of a journal submission.
