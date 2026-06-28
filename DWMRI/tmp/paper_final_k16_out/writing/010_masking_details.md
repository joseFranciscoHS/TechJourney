# 010 — Hybrid RGS Masking Details and Angular-Only Clarification

## Context for the LLM

You are helping complete the **Hybrid RGS Formulation** subsection (§2.2) of a research paper on self-supervised DW-MRI denoising. Two TODOs need resolution:
1. Line 89: Specify mask probability, replacement signal, patch size, loss penalty, and replacement mode
2. Line 167: Clarify whether Angular-Only ablation uses all non-target gradients or matched-channel-budget subset

## Files to attach

| File | Purpose |
|------|---------|
| `paper/Sepulveda_dwmri_restormer.tex` | Current manuscript with TODOs at lines 89 and 167 |
| `src/drcnet_hybrid_rgs/config.yaml` | Configuration showing mask_p, patch sizes, etc. |
| `src/drcnet_hybrid_rgs/fit.py` | Loss implementation (masked MSE) |

## Prompt

Write two text blocks to resolve the TODOs:

### Block 1: Masking specification (line 89)

Replace the TODO at line 89 with this specification paragraph (integrate into existing text flow):

**Technical specifications:**
- **Mask probability p**: 0.3 (Bernoulli sampling per voxel; P(occluded) = 0.3, P(visible) = 0.7)
- **Replacement signal r**: Zero (masked voxels set to 0 via element-wise multiplication)
- **Patch size**: Progressive schedule — Stage 1: 32³, Stage 2: 24³, Stage 3: 16³ voxels
- **Loss penalty ρ**: Squared error (L2); no robust penalty applied
- **Final replacement mode**: Zero replacement throughout training and inference

**Implementation detail**: The mask is applied only to the target channel at index K−1, while context channels remain unmasked. The masked target input is formed as:

```
tilde_y_t = y_t * (1 - m)
```

where `m ∈ {0,1}^(Nx × Ny × Nz)` with `m_i = 1` indicating an occluded (supervised) site.

### Block 2: Angular-Only clarification (line 167)

Insert this clarification after the Angular-Only results paragraph in the objective-controlled ablation section:

**Clarification on Angular-Only channel budget**: The DRCNet Angular-Only condition reported in Table~\ref{tab:objective_controlled_ablation} uses K−1 = 15 randomly sampled context gradients as input, with no target channel present. This is a **matched-channel-budget ablation** designed to isolate the contribution of target-channel spatial masking while keeping the total input dimensionality comparable to Hybrid RGS (K=16). The condition does not represent a full neural Patch2Self replacement using all G−1 = 59 non-target gradients, which would require a different network architecture and is left for future work. The narrow channel budget explains the severe degradation to 10.87 dB PSNR-ROI: inferring each target voxel from only 15 randomly selected diffusion directions, without access to the target's own spatial context, fails to capture high-frequency anatomical detail.

## Expected output

Two LaTeX text blocks:
1. A concise technical specification (80-120 words) stating all five parameters for line 89
2. A clarification paragraph (120-150 words) for line 167 explaining the Angular-Only channel budget and its interpretation

Both should use existing manuscript notation and integrate smoothly into the surrounding text.
