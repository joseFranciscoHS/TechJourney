# 025 — Restormer-3D Larger-Backbone Capacity Ablation

## Context for the LLM

You are helping revise a research paper on **self-supervised denoising of Diffusion-Weighted MRI (DWI)** using a method called **Hybrid RGS** (Random Gradient Subset). The paper evaluates neural backbone architectures under the same Hybrid RGS training objective: random gradient subset sampling, Bernoulli target masking, and masked MSE loss.

The current manuscript [`paper/Sepulveda_dwmri_restormer.tex`](../../../paper/Sepulveda_dwmri_restormer.tex) reports **Restormer-3D** as a lightweight 3-level 3D transformer backbone (`restormer_hybrid_rgs.model.Restormer3D`) with **~0.18M parameters** (`dim=12`, `num_blocks=[1,2,2]`, `heads=[1,2,2]`, `ffn_expansion_factor=1.5`).

**What has changed:** A new ablation arm evaluates whether **scaling the same Restormer-3D backbone to ~2M parameters** (roughly 11x capacity) improves D-Brain denoising under the identical Hybrid RGS protocol. The larger model is obtained purely through **configuration** (no architecture change): a wider base dimension plus deeper transformer stacks and a larger feed-forward expansion. Capacity is added preferentially at the low-resolution (latent) level, which is parameter-dense but memory-cheap.

This prompt asks you to:

1. **Add a methods sentence/paragraph** noting the larger-backbone Restormer-3D variant and its exact configuration.
2. **Clarify in the capacity/dimensionality ablation** that Restormer-3D is now evaluated at two capacities (baseline ~0.18M and large ~2M), isolating the effect of backbone size from architecture and dimensionality.
3. **Fill in the results** for the new large arm once the rerun registry is available (see placeholder table below).

---

## Configuration difference: baseline Restormer-3D -> large Restormer-3D

Both models are the **same** `Restormer3D` class (3-level U-Net topology, volumetric MDTA/GDFN blocks, strided `Conv3d` / `ConvTranspose3d` resampling). Only capacity hyperparameters differ. All training/inference/eval settings are identical (K=16 RGS, G=60, `target_channel=15`, `mask_p=0.3`, progressive patches 32/24/16, subset 0.6, seed 91021, per-volume rescale + clip, `metrics_roi_threshold=0.02`, DTI on).

| Hyperparameter          | Baseline Restormer-3D | Large Restormer-3D              |
| ----------------------- | --------------------- | ------------------------------- |
| `dim`                   | 12                    | **24**                          |
| `num_blocks`            | `[1,2,2]`             | **`[2,4,10]`**                  |
| `heads`                 | `[1,2,2]`             | **`[1,2,4]`**                   |
| `ffn_expansion_factor`  | 1.5                   | **2.66**                        |
| `num_refinement_blocks` | 2                     | **4**                           |
| Parameters              | 177,883 (~0.18M)      | **~2.0M** (fill exact from run) |

Rationale: channel widths per level are `dim / 2*dim / 4*dim` = `24 / 48 / 96`, so `heads=[1,2,4]` divides each level cleanly. The bulk of the added parameters sits in the 10-block latent stack at 96 channels (1/4 spatial resolution), which grows capacity while keeping peak activation memory moderate.

---

## Files to attach

Paths are relative to `DWMRI/`.

| File                                                                      | Purpose                                                |
| ------------------------------------------------------------------------- | ------------------------------------------------------ |
| `paper/Sepulveda_dwmri_restormer.tex`                                     | Current manuscript to patch                            |
| `src/restormer_hybrid_rgs/model.py`                                       | `Restormer3D` (same class, both capacities)            |
| `src/restormer_hybrid_rgs/config.yaml`                                    | Baseline config (`dim=12`, `num_blocks=[1,2,2]`, etc.) |
| `experiments/rerun_k16_restormer3d_large.sh`                              | Run provenance (baseline + large arms)                 |
| `tmp/paper_final_k16_restormer3d_large/registry.jsonl`                    | **Results registry** (available after rerun)           |
| `tmp/paper_final_k16_restormer3d_large/paper_tables/registry_summary.csv` | Flat metrics (available after rerun)                   |

---

## Prompt

You are revising `paper/Sepulveda_dwmri_restormer.tex`. Perform the following tasks. Produce LaTeX snippets that can be inserted or substituted directly.

### Task 1 — Methods note: larger-backbone Restormer-3D

In the architecture/experimental-setup section, add a short paragraph that:

- States that Restormer-3D is evaluated at two capacities under the identical Hybrid RGS protocol: the baseline (~0.18M) and a larger variant (~2M).
- Gives the large config (`dim=24`, `num_blocks=[2,4,10]`, `heads=[1,2,4]`, `ffn_expansion_factor=2.66`, `num_refinement_blocks=4`).
- Emphasizes this is a pure capacity scaling of the _same_ architecture (no change to blocks, resampling, or I/O), added preferentially at the latent level.
- States the approximate parameter count from the run.

Target: ~90–120 words, academic/concise, IEEE-style. Use `\cite{restormer}` for Zamir et al.

### Task 2 — Clarify the capacity/dimensionality ablation

In the ablation setup paragraph, state that the design now separates:

- (A) Hybrid RGS training vs alternatives,
- (B) **backbone capacity** (Restormer-3D ~0.18M vs ~2M, same architecture),
- (C) volumetric 3D vs 2D per-slice processing.

Note that the two Restormer-3D capacities share seed, subset fraction, sampling, masking, and evaluation, so any metric difference reflects capacity alone.

### Task 3 — Results table update (fill after run)

Add a capacity block (or extend the existing Restormer table) with one row per arm:

| Model                        | Params (M) | PSNR-ROI (dB) | SSIM-ROI | FA-MAE | MD-MAE |
| ---------------------------- | ---------- | ------------- | -------- | ------ | ------ |
| Restormer-3D (baseline)      | [FILL]     | [FILL]        | [FILL]   | [FILL] | [FILL] |
| **Restormer-3D (large ~2M)** | [FILL]     | [FILL]        | [FILL]   | [FILL] | [FILL] |

**PLACEHOLDER — fill after running `rerun_k16_restormer3d_large.sh`:**

```
Results location after run:
  tmp/paper_final_k16_restormer3d_large/paper_tables/registry_summary.csv
  tmp/paper_final_k16_restormer3d_large/registry.jsonl

Job ID for baseline:   restormer_dbrain_3d_baseline_k16
Job ID for large ~2M:  restormer_dbrain_3d_large2M_k16
```

Do not invent numeric values. Leave `[FILL]` placeholders and add a LaTeX `\TODO{}` note in the source.

---

## Expected output

1. A LaTeX paragraph for the methods/setup section (Task 1).
2. A revised ablation-setup paragraph separating training / capacity / dimensionality (Task 2).
3. A LaTeX `tabular` fragment with the baseline and large Restormer-3D rows and `\TODO{}` placeholders for the metrics (Task 3).
