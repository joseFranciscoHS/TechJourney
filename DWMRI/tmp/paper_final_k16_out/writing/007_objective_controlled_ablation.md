# 007 — Objective-Controlled DRCNet Ablation

## Context for the LLM

You are helping write a research paper on **self-supervised denoising of Diffusion-Weighted MRI (DWI)** using a method called **Hybrid RGS** (Random Gradient Subset). The core idea: the model receives K randomly sampled gradient volumes as input channels and reconstructs one target volume using a blind-spot (J-invariant) loss. Two backbone architectures are evaluated: **DRCNet** (gated 3D CNN) and **Restormer** (3D transformer).

This prompt asks you to draft the **objective-controlled ablation subsection**, which is the **central ablation of the paper**. It isolates the contribution of each component of the Hybrid RGS training objective — angular context, target-channel spatial masking, and random gradient subset sampling — while keeping the backbone architecture (DRCNet) fixed. This definitively answers whether the improvement comes from the training framework or merely from network capacity.

**Key decisions already made:**
- All four conditions use the **same DRCNet backbone**, same optimizer, same learning rate, same training budget, same D-Brain dataset with Rician noise σ=0.1, and the same evaluation protocol.
- This is a **DRCNet-only** ablation (not Restormer). The point is to hold the architecture constant and vary only the self-supervised objective.
- The four conditions are:
  1. **DRCNet Angular-Only** (Patch2Self-style): K−1 context gradients as input, target excluded from input, no spatial masking. Loss computed over the full target volume.
  2. **DRCNet Spatial-Only** (Self2Self/MD-S2S-style): only the target volume (masked) as input, no angular context. Loss computed only on masked voxels.
  3. **DRCNet Sequential Hybrid**: target masked + K−1 sequential (deterministic) context gradients. Loss on masked voxels.
  4. **DRCNet Hybrid RGS** (proposed): target masked + K−1 random context gradients. Loss on masked voxels.
- The Angular-Only condition performs dramatically worse in PSNR-ROI (~10.9 dB), which is expected and informative — it shows that excluding the target from input removes crucial local spatial information, even when angular context is available.
- The Spatial-Only condition (~24.5 dB) and Sequential Hybrid (~23.6 dB) are both reasonable but clearly below Hybrid RGS (~26.9 dB).
- FA-MAE differences across conditions are smaller than PSNR differences — discuss this honestly. All conditions preserve tensor structure to a degree, but image-domain quality varies substantially.

## Files to attach

Attach these files when you send this prompt. Paths are relative to `DWMRI/`.

| File | Purpose |
|------|---------|
| `tmp/paper_final_k16_out/paper_tables/paper_metrics_summary.csv` | Complete metrics. For this ablation, extract four rows: (1) path containing `dbrain_angular_only` with `k_input=15` — the Angular-Only condition; (2) path containing `dbrain_spatial_only` with `k_input=1` — the Spatial-Only condition; (3) `sampling_mode=sequential` with `k_input=16` for DRCNet on D-Brain — the Sequential Hybrid condition; (4) `sampling_mode=rgs` with `k_input=16` for DRCNet on D-Brain at sigma=0.1 with `n_context_samples=16, n_preds=12` — the Hybrid RGS condition (use the k_sweep K=16 row or the architecture parity row, not the row with n_context_samples=48 which had inference issues). Key columns: `psnr_roi`, `ssim_roi`, `fa_mae`, `md_mae`. |
| `tmp/paper_final_k16_out/paper_tables/paper_runtime_summary.csv` | Runtime and parameter counts. Filter for `recipe=objective_controlled_ablation` (Angular-Only and Spatial-Only rows) and for the Sequential and main RGS runs. Key columns: `n_params`, `sec_per_volume`, `duration_s`. |
| `experiments/pruebas_faltantes_20260627.md` | Experimental design document (in Spanish). Sections 1 (Objective-Controlled DRCNet Ablation) and 16 (Interpretation) describe the rationale, expected outcomes, and the claim this ablation supports. Read sections 1A–1D for the precise definition of each condition. |
| `src/J_invariance_DWMRI_denoising_report_hybrid_rgs.md` | Technical report. Provides background on the J-invariance framework, the masking scheme, and why angular + spatial coupling is expected to outperform each component alone. |

## Prompt

Write the **Objective-Controlled Ablation** subsection for the Ablation Studies section of the paper. This is the most important ablation — it proves that the contribution of Hybrid RGS is the training objective itself, not the network architecture.

### Structure to produce

#### Motivation (~120-150 words, 1 paragraph)

- The main comparison table (presented earlier in the paper) showed that DRCNet and Restormer trained with Hybrid RGS outperform classical baselines. However, this leaves open whether the improvement is due to the deep learning backbone or the specific self-supervised objective.
- To disentangle these factors, we fix the backbone to DRCNet and systematically vary the training objective along three axes: (a) whether angular context from other gradient directions is used, (b) whether the target volume undergoes spatial blind-spot masking, and (c) whether the angular context is sampled randomly or deterministically.
- This yields four conditions that span the design space of self-supervised DWI denoising objectives.

#### Experimental conditions (~200-250 words, 1-2 paragraphs)

Describe each condition concisely:

1. **DRCNet Angular-Only** — conceptually equivalent to a neural Patch2Self. The network receives K−1 randomly sampled context gradients (excluding the target) and predicts the full target volume. The target is not included in the input, so no spatial masking is needed. The loss is computed over all voxels of the target. This tests whether angular redundancy alone suffices for denoising.

2. **DRCNet Spatial-Only** — conceptually equivalent to a volumetric neural Self2Self or MD-S2S. The network receives only the target volume with Bernoulli-masked voxels and predicts the missing voxels. No other gradient directions are provided. This tests whether local spatial context alone (via blind-spot masking) suffices without angular information.

3. **DRCNet Sequential Hybrid** — combines angular context and spatial masking, but context gradients are selected as a deterministic sliding window rather than random subsets. This isolates the value of randomness in the sampling strategy.

4. **DRCNet Hybrid RGS (proposed)** — the full method: masked target + K−1 randomly sampled context gradients. Loss on masked voxels only.

Emphasize that all four share the same backbone, optimizer, learning rate, training steps, patch size, normalization, and evaluation protocol. The only variable is the training objective.

#### Results and analysis (~250-350 words, 2-3 paragraphs)

1. **Angular-Only performs dramatically worse** (~10.9 dB PSNR-ROI). Discuss why: without the target volume in the input, the network must reconstruct each voxel purely from angular neighbors. While the angular signal carries correlated information, it lacks the high-frequency spatial detail present in the target volume itself. The result is severe blurring. This rules out the pure Patch2Self-style neural approach as competitive with methods that include spatial masking.

2. **Spatial-Only is a strong baseline** (~24.5 dB PSNR-ROI). The blind-spot masking alone provides substantial denoising by exploiting local spatial correlations. Interestingly, Spatial-Only outperforms Sequential Hybrid (~23.6 dB) in PSNR-ROI, suggesting that naively adding angular context in a deterministic window can actually hurt if the fixed context neighborhoods introduce bias. However, Spatial-Only is still 2+ dB below Hybrid RGS — angular context does help, but only when sampled properly.

3. **Hybrid RGS achieves the best results** (~26.9 dB PSNR-ROI). The combination of angular context, spatial masking, and random subset sampling yields the highest image quality. The gain over Sequential Hybrid (~3.3 dB) demonstrates that random sampling provides valuable angular diversity — it acts as implicit data augmentation, exposing the network to varied angular contexts that prevent overfitting to fixed channel neighborhoods.

4. **FA-MAE is relatively stable across conditions** (range ~0.22–0.26). All objectives preserve tensor structure to a similar degree, which suggests that the FA metric is less sensitive to the image-domain quality differences. Discuss this honestly: the main benefit of Hybrid RGS over ablated variants is in image reconstruction fidelity (PSNR/SSIM), while diffusion tensor preservation is comparable. This is consistent with the observation that even noisy data can yield reasonable tensor fits when averaged over a voxel neighborhood.

#### Key claim (1 sentence to weave into the discussion)

> Using the same DRCNet backbone, Hybrid RGS outperforms angular-only, spatial-only, and sequential hybrid objectives, confirming that the improvement stems from the coupled random-gradient-subset and target-masking training scheme rather than from network capacity alone.

### Table to produce

Create a `booktabs` table with `\label{tab:objective_controlled_ablation}` and the following structure:

```
Training Objective | Angular | Masking | Random | PSNR-ROI↑ | SSIM-ROI↑ | FA-MAE↓ | MD-MAE↓ | Time/vol (s)
```

- Use checkmarks (`\checkmark`) and dashes (`--`) for the boolean columns (Angular context, Target masking, Random subset).
- Four rows: Angular-Only, Spatial-Only, Sequential Hybrid, Hybrid RGS.
- Bold the best value in each metric column.
- Caption: "Objective-controlled ablation on D-Brain (Rician σ=0.1). All rows use the same DRCNet backbone, training budget, and evaluation protocol. The ablation isolates angular context, target-channel masking, and random gradient subset sampling."
- Extract exact values from the attached CSV files. For Hybrid RGS, use the K=16 DRCNet result at sigma=0.1 with n_context_samples=16 and n_preds=12 (the standard inference configuration).

### Formatting requirements

- LaTeX source (compilable as a subsection).
- Use `\subsection{Objective-Controlled Ablation}`.
- Use `booktabs` for the table (`\toprule`, `\midrule`, `\bottomrule`).
- Reference as `Table~\ref{tab:objective_controlled_ablation}`.
- Cross-reference the main comparison table: "As shown in Table~\ref{tab:main_comparison}, ..." when motivating the ablation.
- Cross-reference the sampling ablations: "The sequential vs.\ RGS comparison in Table~\ref{tab:sampling_ablations} already suggested ..." if useful for continuity.
- Total target length: **1.5-2 pages** including table (approximately 600-800 words of body text).
- Do not include `\begin{document}` or preamble — just the subsection content.

### What to avoid

- Do not downplay the Angular-Only failure. It is a strong result: it shows that a neural Patch2Self with the same backbone cannot match methods that include the target channel. Frame it as informative, not as a negative.
- Do not hide that Spatial-Only beats Sequential Hybrid. Discuss it: deterministic angular windows may introduce bias, while the spatial blind-spot objective is inherently unbiased.
- Do not overstate the FA-MAE differences. They are small and should be presented honestly. The main story is in PSNR/SSIM.
- Do not claim that this ablation proves RGS is optimal for all architectures. It proves it for DRCNet; the architecture transfer is shown elsewhere in the paper.
- Do not include results from Restormer in this table. This is a single-backbone ablation by design.

## Expected output

A single LaTeX code block containing:
- `\subsection{...}` with the full narrative text (~600-800 words).
- One `table` environment with the objective-controlled ablation results (4 rows, 8 columns).
- Appropriate `\label`, `\cite`, and `\ref` commands.
- No preamble, no `\begin{document}`.
