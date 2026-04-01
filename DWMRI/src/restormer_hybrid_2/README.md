# Restormer-Hybrid: 3D Restormer for DWMRI Denoising

This module implements the **Restormer architecture adapted for 3D volumetric data**, using the **hybrid Multi-Directional Self2Self (MD-S2S)** training methodology for J-invariant denoising of diffusion-weighted MRI.

## Overview

Restormer-Hybrid combines:
1. **Restormer Architecture** (Zamir et al., CVPR 2022) - An efficient transformer for image restoration
2. **3D Adaptation** - All 2D operations converted to 3D for volumetric medical imaging
3. **Hybrid MD-S2S Training** - Self-supervised denoising using J-invariance principles
4. **Progressive Learning** - Optional multi-stage training with increasing patch sizes (16³ → 24³ → 32³) and AMP

### Key Features

- **Self-supervised**: No clean ground truth required for training
- **J-invariant loss**: Prevents identity mapping through masked pixel loss
- **Multi-directional context**: Uses all DWI volumes to denoise each target volume
- **Monte Carlo inference**: Averages multiple masked predictions for robust reconstruction
- **Progressive learning**: Train on small patches first, then larger ones for better structural integrity (e.g. tractography)
- **Sliding-window reconstruction**: Full-volume inference via overlapping patches with Gaussian blending to avoid OOM and seam artifacts
- **Learnable output scale and shift**: Optional trainable scale/shift so outputs match the input data range (avoids "shrunk" reconstructions)

## Architecture

The model uses a **3-level hierarchy** (encoder → latent → decoder) to preserve structural integrity on typical DWMRI volumes (e.g. 128×128×96). A 4th level was avoided to prevent excessive downsampling (8× voxel drop per level in 3D).

```
Input (B, num_vols, D, H, W)
    │
    ▼
┌─────────────────────────┐
│   Patch Embedding 3D    │  Conv3d(3x3x3)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Encoder Level 1–3     │  TransformerBlock3D × num_blocks[i]
│   (with Downsampling)   │  Strided Conv3d(2×2×2) between levels
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Latent Transformer    │  TransformerBlock3D × num_blocks[2]
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Decoder Level 3–1     │  TransformerBlock3D × num_blocks[i]
│   (with Upsampling)     │  ConvTranspose3d(2×2×2) + Skip connections
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Refinement Blocks     │  TransformerBlock3D × num_refinement_blocks
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Output Projection     │  Conv3d(3x3x3) + PReLU/Sigmoid
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Scale & Shift (opt.)  │  out = scale * out + shift (learnable; fixes output range)
└─────────────────────────┘
    │
    ▼
Output (B, 1, D, H, W)
```

### TransformerBlock3D Components

- **LayerNorm3D**: Layer normalization over channel dimension for 5D tensors (B, C, D, H, W)
- **Attention3D**: Multi-DConv Head Transposed Self-Attention (channel-wise C×C attention; linear in spatial size)
- **FeedForward3D**: Gated-DConv Feed-Forward Network with depthwise 3D convolutions

### Output scale and shift

When `scale_and_shift` is enabled (default: true), two learnable scalars are applied after the output projection: `out = output_scale * out + output_shift`. This lets the model match the data range (e.g. [0, 1] normalized DWI) and avoids systematically low ("shrunk") reconstructions. Disable by setting `scale_and_shift: false` in config (under `train` or where the model is built).

## Training Methodology

### Hybrid MD-S2S Approach (Scheme 2)

1. **Input**: All volumes including target volume with pixel-level Bernoulli mask applied
2. **Mask**: Random spatial mask (default 30% of pixels masked)
3. **Loss**: MSE computed only on masked pixels (J-invariant)
4. **Inference**: Monte Carlo averaging over multiple random masks

```python
# Training: mask only the target volume
x_masked = all_volumes.clone()
x_masked[target_vol] = x_masked[target_vol] * mask

# Loss: only on masked pixels
loss = MSE(prediction, target) * (1 - mask) / sum(1 - mask)
```

### Progressive Learning (Optional)

When `progressive.enabled: true` in `config.yaml`, training runs in stages with increasing patch sizes. This improves local noise statistics first, then structural boundaries, then global context (e.g. for tractography). **AMP (Automatic Mixed Precision)** is used to fit larger patches within GPU memory.

| Stage | Patch size | Batch | Epochs | Goal |
|-------|------------|-------|--------|------|
| 1 | 16³ | 4 | 10 | Local noise statistics |
| 2 | 24³ | 2 | 10 | Structural boundaries |
| 3 | 32³ | 1 | 10 | Global context |

Checkpoints are saved per stage; the best checkpoint from the final stage is copied to the main checkpoint directory for reconstruction.

### Reconstruction (Sliding Window)

Full-volume inference is done via **sliding-window patches** to avoid OOM on large volumes (e.g. 128×128×96):

1. Volume is tiled with overlapping patches (`patch_size`, `overlap` in config).
2. Each patch is passed through the model; outputs are accumulated with **3D Gaussian blending weights** to avoid seam/chessboard artifacts.
3. Results are normalized by the weight sum and optionally averaged over `n_preds` masked predictions.

Reconstruction `patch_size` and `overlap` should match the final training stage (e.g. 32 and 16 for 50% overlap) when using progressive learning.

## Usage

### Training and reconstruction

```bash
cd /path/to/DWMRI/src
python -m restormer_hybrid.run
```

From Python (e.g. Jupyter):

```python
from restormer_hybrid.run import main
main("dbrain")  # train=True, reconstruct=True, generate_images=True
main("dbrain", train=False, reconstruct=True)  # inference only
```

### Configuration

Edit `config.yaml` (under `dbrain` or `stanford`) to customize:

```yaml
model:
  in_channel: 10
  out_channel: 1
  dim: 24                 # Smaller = fewer params and faster (e.g. 24 or 32)
  num_blocks: [1, 2, 2]   # 3 levels; fewer blocks = faster
  heads: [1, 2, 2]        # Must match num_blocks length
  num_refinement_blocks: 1
  ffn_expansion_factor: 1.5
  output_activation: "prelu"  # or "sigmoid" for [0,1]

train:
  num_epochs: 30          # Total when progressive; else single-stage epochs
  batch_size: 2           # Default; overridden by progressive stages
  use_amp: true           # Mixed precision for memory savings
  scale_and_shift: true   # Learnable output scale/shift (recommended; fixes shrunk range)
  mask_p: 0.3
  learning_rate: 0.00045  # Tune with batch size and stages
  progressive:
    enabled: true
    stages:
      - patch_size: 16
        batch_size: 4     # Increase if GPU memory allows
        epochs: 10
        step: 4
      - patch_size: 24
        batch_size: 2
        epochs: 10
        step: 6
      - patch_size: 32
        batch_size: 1
        epochs: 10
        step: 8

reconstruct:
  patch_size: 32   # Match final progressive stage
  overlap: 16      # 50% overlap for Gaussian blending
  n_preds: 4       # Fewer = faster; more = smoother

data:
  patch_size: 16   # Used when progressive disabled
  step: 4
  patch_filter_method: "threshold"
```

## Memory Considerations

3D volumetric data and transformer blocks are memory-heavy. You can reduce parameters and speed up training by lowering `dim`, `num_blocks`, `num_refinement_blocks`, and `ffn_expansion_factor`. Recommended ranges:

| Parameter | 2D Restormer | 3D (heavier) | 3D (lighter, faster) |
|-----------|--------------|---------------|----------------------|
| `dim` | 48 | 32 | 24 |
| `num_blocks` | [4,6,6,8] | [2,2,4] | [1,2,2] |
| `num_refinement_blocks` | — | 2 | 1 |
| `ffn_expansion_factor` | — | 2.0 | 1.5 |
| `batch_size` | 8 | 2 | 2–4 (per stage) |
| `patch_size` | 64 | 16 | 16→24→32 (progressive) |
| Reconstruction | Full image | Sliding window (patch_size, overlap) | Same; match final stage |

Enable `use_amp: true` and `scale_and_shift: true` for stable training and output range.

## File Structure

```
restormer_hybrid/
├── __init__.py          # Module initialization
├── model.py             # Restormer3D architecture (3-level hierarchy)
├── data.py              # Dataset and patch extraction (MD-S2S masking)
├── fit.py               # Training loop (J-invariant loss, AMP support)
├── reconstruction.py    # Sliding-window inference with Gaussian blending
├── run.py               # Main orchestrator (progressive + standard training)
├── config.yaml          # Hyperparameters and progressive stages
├── ideas.md             # Notes on progressive learning and hierarchy
└── README.md            # This file
```

## References

- **Restormer**: Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration", CVPR 2022
- **Self2Self**: Quan et al., "Self2Self With Dropout: Learning Self-Supervised Denoising From Single Image", CVPR 2020
- **Patch2Self**: Fadnavis et al., "Patch2Self: Denoising Diffusion MRI with Self-Supervised Learning", NeurIPS 2020
- **J-Invariance**: Batson & Royer, "Noise2Self: Blind Denoising by Self-Supervision", ICML 2019

## Citation

If you use this code, please cite the relevant papers and this implementation.
