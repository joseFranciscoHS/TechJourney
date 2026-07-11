# 024 — Restormer-2D Hybrid RGS: Architecture Description and 3D-vs-2D Ablation Update

## Context for the LLM

You are helping revise a research paper on **self-supervised denoising of Diffusion-Weighted MRI (DWI)** using a method called **Hybrid RGS** (Random Gradient Subset). The paper evaluates neural backbone architectures under the same Hybrid RGS training objective: random gradient subset sampling, Bernoulli target masking, and masked MSE loss.

The current manuscript [`paper/Sepulveda_dwmri_restormer.tex`](../../paper/Sepulveda_dwmri_restormer.tex) evaluates three architecture families:

1. **DRCNet-3D** — gated 3D CNN backbone
2. **Restormer-3D** — 3-level 3D transformer backbone (`restormer_hybrid_rgs.model.Restormer3D`)
3. **Lightweight 2D baselines** — described as "two-dimensional per-slice feature processing"

The 2D baseline referred to in the manuscript is currently `Res-CNN-2D`: a minimal residual CNN (embed → N residual blocks → output; no attention, no hierarchy). Its code lives in `src/restormer_hybrid_rgs/model2d.ResCNN2D` (previously misnamed `Restormer2D`).

**What has changed:** A new 2D backbone has been implemented — `Restormer2D` in `src/restormer_hybrid_rgs/restormer_arch_2d.py`. This is a **capacity-matched** port of the original Zamir et al. (CVPR 2022) Restormer with the same Hybrid RGS I/O adaptations as Restormer3D, but using 2D MDTA/GDFN blocks and PixelUnshuffle/PixelShuffle resampling instead of 3D convolutions. This enables a clean 2D-vs-3D comparison *within the same transformer architecture family*.

This prompt asks you to:

1. **Add a methods paragraph** describing Restormer-2D Hybrid RGS and how it differs from both the original Zamir Restormer and from Restormer-3D.
2. **Clarify in the 3D-vs-2D ablation section** that the ablation now includes three Restormer-family arms: Res-CNN-2D (minimal CNN), Restormer-2D (true transformer, 2D), and Restormer-3D (true transformer, 3D).
3. **Fill in the results** for the new Restormer-2D arm once the rerun registry is available (see placeholder table below).

---

## Architecture differences: original Restormer → Restormer-2D Hybrid RGS

### What is unchanged from Zamir et al.

- **MDTA block:** Multi-DConv Head Transposed Self-Attention (`Attention` class): channel-wise attention with `qkv_dwconv`, normalized query/key, temperature-scaled softmax.
- **GDFN block:** Gated-Dconv Feed-Forward Network (`FeedForward` class): `project_in → dwconv → gated-GELU → project_out`.
- **Transformer block structure:** `x = x + MDTA(LN(x)); x = x + GDFN(LN(x))`.
- **Resampling:** `Downsample` = `Conv2d + PixelUnshuffle(2)`; `Upsample` = `Conv2d + PixelShuffle(2)`. These parameter-light channel-space rearrangements are preserved (contrast: Restormer3D uses strided `Conv3d` / `ConvTranspose3d` due to absence of native 3D pixel-shuffle in PyTorch).
- **Patch embedding:** `OverlapPatchEmbed` = `Conv2d 3×3`.

### What differs (Hybrid RGS adaptations)

| Aspect | Original Restormer | Restormer-2D Hybrid RGS |
|--------|-------------------|------------------------|
| Hierarchy depth | 4 levels, 3 downsamples | **3 levels, 2 downsamples** (matches Restormer3D) |
| Capacity (`dim`) | 48 | **12** (matches Restormer3D config) |
| `num_blocks` | `[4,6,6,8]` | **`[1,2,2]`** |
| `heads` | `[1,2,4,8]` | **`[1,2,2]`** |
| `ffn_expansion_factor` | 2.66 | **1.5** |
| `num_refinement_blocks` | 4 | **2** |
| `inp_channels` | 3 (RGB) or 6 (dual-pixel) | **K** (RGS context gradients + masked target at index K−1) |
| `out_channels` | 3 (RGB) | **1** (single denoised volume) |
| Global input residual | `out + inp_img` (enabled; channels match) | **Removed** (K→1 makes it undefined) |
| Output activation | None (linear) | **PReLU** + learned `scale_and_shift` affine |
| Spatial size constraint | Requires H,W ÷ 4 | Same; `forward()` reflect-pads to ÷4 and crops back |

### What differs from Restormer-3D (same repo)

| Aspect | Restormer-3D | Restormer-2D |
|--------|-------------|-------------|
| Spatial operations | 3D (`Conv3d`, volumetric MDTA) | 2D (`Conv2d`, per-slice MDTA) |
| Resampling | Strided `Conv3d` / `ConvTranspose3d` | `PixelUnshuffle(2)` / `PixelShuffle(2)` |
| Input spatial dims | (D, H, W) patches — volumetric | (H, W) axial slices — no through-plane context |
| Parameters (K=16) | ~0.178M | ~[FILL FROM RUN] M |

### What differs from Res-CNN-2D

Res-CNN-2D has no attention, no U-Net hierarchy, and no gating: it is a flat embed → residual-blocks → output stack. Its purpose is to test the *lower bound* of backbone capacity under Hybrid RGS. Restormer-2D tests whether a *transformer* backbone provides added value at matched 3D capacity when restricted to 2D per-slice processing.

---

## Files to attach

Attach these files when you send this prompt. Paths are relative to `DWMRI/`.

| File | Purpose |
|------|---------|
| `paper/Sepulveda_dwmri_restormer.tex` | Current manuscript to patch |
| `src/restormer_hybrid_rgs/restormer_arch_2d.py` | Restormer-2D implementation (new) |
| `src/restormer_hybrid_rgs/model2d.py` | ResCNN2D implementation (renamed) |
| `src/restormer_hybrid_rgs/model.py` | Restormer3D for comparison |
| `src/restormer_hybrid_rgs/config.yaml` | Shared config (`dim=12`, `num_blocks=[1,2,2]`, etc.) |
| `experiments/rerun_k16_restormer2d_ablation.sh` | Run provenance |
| `tmp/paper_final_k16_restormer2d_ablation/registry.jsonl` | **Results registry** (available after rerun) |
| `tmp/paper_final_k16_restormer2d_ablation/paper_tables/registry_summary.csv` | Flat metrics (available after rerun) |

---

## Prompt

You are revising `paper/Sepulveda_dwmri_restormer.tex`. Perform the following three tasks. Produce LaTeX snippets that can be inserted or substituted directly.

### Task 1 — New methods paragraph: Restormer-2D Hybrid RGS

Under `\subsubsection{Lightweight 2D Baselines for the Capacity-Dimensionality Ablation}` (currently lines ~254–261 in the manuscript), **add a paragraph after the existing Res-CNN-2D description** that:

- Introduces Restormer-2D as a capacity-matched 2D port of the original Restormer.
- States the 3-level hierarchy (`dim=12`, `num_blocks=[1,2,2]`, `heads=[1,2,2]`, `ffn_expansion_factor=1.5`).
- Notes that MDTA/GDFN blocks and PixelUnshuffle/PixelShuffle resampling are structurally identical to the original Zamir et al. architecture.
- Explains that the only Hybrid RGS adaptations are: (1) `inp_channels=K` / `out_channels=1`; (2) removal of the global input residual (undefined for K→1); (3) addition of learned `scale_and_shift` affine + PReLU output activation.
- Contrasts with Restormer3D: Restormer-2D operates per axial slice with no through-plane context; Restormer3D uses `Conv3d` and volumetric MDTA.
- States the approximate parameter count from the run.

Target: ~120–150 words, academic/concise, IEEE-style. Use `\cite{restormer}` for Zamir et al.

### Task 2 — Clarify Res-CNN-2D vs Restormer-2D in the ablation setup

In the experimental setup paragraph for the 3D-vs-2D ablation (Section 4.6 or similar), replace any wording that implies the 2D Restormer variant uses per-slice transformer attention with accurate text stating:

- The 2D ablation now includes **three** Restormer-family arms: Res-CNN-2D (lightweight CNN, minimal capacity), Restormer-2D (true MDTA/GDFN, 2D per-slice, capacity-matched to Restormer3D), and Restormer-3D (volumetric MDTA/GDFN).
- Res-CNN-2D and Restormer-2D are both trained slice-wise (axial); neither has through-plane context.
- This three-arm design separates the contribution of (A) Hybrid RGS training, (B) architectural capacity, and (C) volumetric 3D processing.

### Task 3 — Results table update (fill after run)

Extend `\tab:3d_vs_2d` (or equivalent) to include the Restormer-2D row. The table should have the following structure with one row per arm:

| Model | Params (M) | PSNR-ROI (dB) | SSIM-ROI | FA-MAE | MD-MAE |
|-------|-----------|--------------|----------|--------|--------|
| Res-CNN-2D | [existing] | [existing] | [existing] | [existing] | [existing] |
| **Restormer-2D** | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| Restormer-3D | [existing] | [existing] | [existing] | [existing] | [existing] |

**PLACEHOLDER — fill after running `rerun_k16_restormer2d_ablation.sh`:**

```
Results location after run:
  tmp/paper_final_k16_restormer2d_ablation/paper_tables/registry_summary.csv
  tmp/paper_final_k16_restormer2d_ablation/registry.jsonl

Job ID for Restormer-2D:  restormer_dbrain_restormer2d_rgs_k16_ablation
Job ID for Res-CNN-2D:    restormer_dbrain_2d_rgs_k16_ablation
```

Do not invent numeric values. Leave `[FILL]` placeholders and add a LaTeX `\TODO{}` note in the source.

---

## Expected output

1. A LaTeX paragraph for the methods section (Task 1).
2. A revised experimental-setup paragraph for Section 4.6 (Task 2).
3. A LaTeX `tabular` fragment extending `tab:3d_vs_2d` with the Restormer-2D row and `\TODO{}` placeholders for the new metrics (Task 3).
