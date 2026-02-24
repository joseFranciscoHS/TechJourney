# Restormer-Hybrid: 3D Restormer for DWMRI Denoising

This module implements the **Restormer architecture adapted for 3D volumetric data**, using the **hybrid Multi-Directional Self2Self (MD-S2S)** training methodology for J-invariant denoising of diffusion-weighted MRI.

## Overview

Restormer-Hybrid combines:
1. **Restormer Architecture** (Zamir et al., CVPR 2022) - An efficient transformer for image restoration
2. **3D Adaptation** - All 2D operations converted to 3D for volumetric medical imaging
3. **Hybrid MD-S2S Training** - Self-supervised denoising using J-invariance principles

### Key Features

- **Self-supervised**: No clean ground truth required for training
- **J-invariant loss**: Prevents identity mapping through masked pixel loss
- **Multi-directional context**: Uses all DWI volumes to denoise each target volume
- **Monte Carlo inference**: Averages multiple masked predictions for robust reconstruction

## Architecture

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
│   Encoder Level 1-4     │  TransformerBlock3D × num_blocks[i]
│   (with Downsampling)   │  Conv3d(stride=2) between levels
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Latent Transformer    │  TransformerBlock3D × num_blocks[3]
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Decoder Level 4-1     │  TransformerBlock3D × num_blocks[i]
│   (with Upsampling)     │  ConvTranspose3d(stride=2) + Skip connections
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
Output (B, 1, D, H, W)
```

### TransformerBlock3D Components

- **LayerNorm3D**: Layer normalization adapted for 5D tensors
- **Attention3D**: Multi-DConv Head Transposed Self-Attention (channel-wise attention)
- **FeedForward3D**: Gated-DConv Feed-Forward Network with depthwise 3D convolutions

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

## Usage

### Training

```bash
cd /path/to/DWMRI/src
python -m restormer_hybrid.run
```

### Configuration

Edit `config.yaml` to customize:

```yaml
model:
  in_channel: 10          # Number of input volumes
  out_channel: 1          # Output channels (single denoised volume)
  dim: 32                 # Base feature dimension
  num_blocks: [2,2,2,4]   # Transformer blocks per level
  heads: [1,2,4,8]        # Attention heads per level
  ffn_expansion_factor: 2.0

train:
  num_epochs: 10
  batch_size: 4           # Reduced for 3D memory
  mask_p: 0.3             # Bernoulli mask probability
  learning_rate: 0.0015

data:
  patch_size: 16          # Smaller for 3D memory
  patch_filter_method: "threshold"  # Filter background patches
```

## Memory Considerations

3D volumetric data significantly increases memory usage compared to 2D. Recommended adjustments:

| Parameter | 2D Restormer | 3D Restormer |
|-----------|--------------|--------------|
| `dim` | 48 | 32 |
| `num_blocks` | [4,6,6,8] | [2,2,2,4] |
| `batch_size` | 8 | 4 |
| `patch_size` | 64 | 16 |

## File Structure

```
restormer_hybrid/
├── __init__.py          # Module initialization
├── model.py             # Restormer3D architecture
├── data.py              # Dataset and patch extraction (MD-S2S masking)
├── fit.py               # Training loop (J-invariant loss)
├── reconstruction.py    # Monte Carlo inference
├── run.py               # Main orchestrator
├── config.yaml          # Hyperparameters
└── README.md            # This file
```

## References

- **Restormer**: Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration", CVPR 2022
- **Self2Self**: Quan et al., "Self2Self With Dropout: Learning Self-Supervised Denoising From Single Image", CVPR 2020
- **Patch2Self**: Fadnavis et al., "Patch2Self: Denoising Diffusion MRI with Self-Supervised Learning", NeurIPS 2020
- **J-Invariance**: Batson & Royer, "Noise2Self: Blind Denoising by Self-Supervision", ICML 2019

## Citation

If you use this code, please cite the relevant papers and this implementation.
