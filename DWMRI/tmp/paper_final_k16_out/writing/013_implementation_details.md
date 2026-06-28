# 013 — Implementation Details

## Context for the LLM

You are completing the **Implementation Details** subsection (likely §3.6 or an appendix) of a research paper on self-supervised DW-MRI denoising. The TODO at line 359 requests a comprehensive specification of training hyperparameters, hardware, and inference settings.

## Files to attach

| File | Purpose |
|------|---------|
| `paper/Sepulveda_dwmri_restormer.tex` | Current manuscript with TODO at line 359 |
| `src/drcnet_hybrid_rgs/config.yaml` | Full training configuration |
| `src/restormer_hybrid_rgs/config.yaml` | Restormer-specific settings |

## Prompt

Write the **Implementation Details** subsection to replace the TODO at line 359. This section should enable full reproducibility of the experiments.

### Content outline

**Training Configuration**

All models are trained with the Adam optimizer using a base learning rate of $4.5 \times 10^{-4}$ without learning rate scheduling. Training uses automatic mixed precision (AMP) to reduce memory consumption on the available GPU hardware. Random seeds are fixed at 42 for PyTorch, NumPy, and the dataset's RGS gradient sampling to ensure reproducible training patch generation; however, GPU operations remain nondeterministic for performance reasons (`reproducible: false` setting).

**Progressive Training Schedule**

Both DRCNet and Restormer use a three-stage progressive training protocol with decreasing patch sizes:

| Stage | Patch Size | Stride | Batch Size (DRCNet) | Batch Size (Restormer) | Epochs |
|-------|-----------|--------|---------------------|------------------------|--------|
| 1 | 32³ voxels | 8 | 128 | 64 | 120 |
| 2 | 24³ voxels | 6 | 256 | 128 | 120 |
| 3 | 16³ voxels | 4 | 256 | 256 | 120 |

Total training: **360 epochs** for D-Brain; **900 epochs** (300 per stage) for Stanford HARDI. Each stage trains from scratch with a new dataset constructed at the specified patch resolution. The final-stage checkpoint with the lowest epoch-averaged training loss is selected for evaluation.

**Data Augmentation**

No explicit geometric augmentation (flips, rotations, elastic deformations) is applied. Implicit stochasticity arises from (i) random gradient subset sampling in RGS mode, (ii) independent Bernoulli spatial masks per training sample, and (iii) strided patch extraction covering diverse spatial locations. Background patches where $\max(\text{patch}) \leq 0.02$ are filtered during dataset construction.

**Normalization**

Input volumes are normalized per-volume to $[0,1]$ via min-max scaling before training. Reconstruction outputs are optionally rescaled back to $[0,1]$ per-volume before computing metrics (`rescale_to_01: true`, `clip_to_range: true`).

**Inference Settings**

Reconstruction uses Monte Carlo averaging over multiple stochastic contexts and spatial masks:
- **D-Brain**: $N_c = 16$ context samples, $N_p = 12$ mask draws per context
- **Stanford HARDI**: $N_c = 16$, $N_p = 23$
- **Micro-batching**: Mask predictions are computed in chunks of size 4 (`pred_chunk_size: 4`) to manage GPU memory

Restormer additionally uses overlapping patch-based tiled inference with Gaussian weighting (`patch_size: 32`, `overlap: 1`) for full-volume reconstruction.

**Hardware**

All experiments were conducted on a single **NVIDIA Tesla T4 GPU** (15 GB VRAM) with CUDA 12.2 and PyTorch 2.x. Training time for a full 360-epoch D-Brain run is approximately 8-12 hours depending on the model; inference time per target volume ranges from 4.7 s (DRCNet-2D) to 130 s (Restormer-3D) with the specified Monte Carlo sampling settings.

**Checkpoint Selection**

The checkpoint with the **lowest epoch-averaged training loss** across all stages is retained for evaluation (`best_loss_checkpoint.pth`). Validation-set early stopping is not used because the self-supervised setting does not have access to clean validation targets during training.

**Code Availability**

Training and inference implementations: `src/drcnet_hybrid_rgs/` and `src/restormer_hybrid_rgs/`. Configuration files define all hyperparameters in YAML format. Reproducibility note: exact bitwise reproducibility across GPU types is not guaranteed due to cuDNN nondeterminism, but results should be numerically stable within typical run-to-run variance.

## Expected output

A LaTeX subsection (400-500 words) structured as:
1. **Training** (optimizer, LR, AMP, seeds)
2. **Progressive schedule** (table of stages)
3. **Augmentation** (implicit only, no geometric)
4. **Normalization** (per-volume min-max)
5. **Inference** (Nc, Np, chunking)
6. **Hardware** (GPU model, VRAM, timing)
7. **Checkpoint rule** (best training loss)
8. **Code** (pointers to implementation)

Use clear section headings or bold labels. Include the stage table in LaTeX `booktabs` format.
