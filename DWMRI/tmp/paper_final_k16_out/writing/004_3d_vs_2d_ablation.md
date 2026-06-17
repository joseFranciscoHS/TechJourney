# 004 — 3D vs 2D Convolutions: Critical Ablation

## Context for the LLM

You are helping write a research paper on **self-supervised denoising of Diffusion-Weighted MRI (DWI)** using a method called **Hybrid RGS** (Random Gradient Subset). The core idea: the model receives K randomly sampled gradient volumes as input channels and reconstructs one target volume using a blind-spot (J-invariant) loss. Two backbone architectures are evaluated: **DRCNet** (gated 3D CNN) and **Restormer** (3D transformer).

This prompt asks you to draft the **3D vs 2D ablation subsection**, which is **critical for reviewers**. The question: Does the performance improvement come from using 3D convolutions (volumetric context across slices), or solely from the Hybrid RGS training scheme itself? This ablation isolates the architectural contribution from the algorithmic contribution.

**Key decisions already made:**
- This is the most important ablation for defending the paper's claims. Without it, reviewers can attribute all gains to the training scheme and dismiss the architecture choice.
- The 2D variant processes each axial slice independently (Conv2D, no z-connectivity), using the same Hybrid RGS training scheme (K=16, RGS sampling, Bernoulli masking).
- Both architectures (DRCNet and Restormer) are evaluated in 2D mode.
- Results are on **D-Brain σ=0.1 only** (controlled comparison with GT).

## Files to attach

Attach these files when you send this prompt. Paths are relative to `DWMRI/`.

| File | Purpose |
|------|---------|
| `tmp/paper_final_k16_out/paper_tables/paper_metrics_summary.csv` | Complete metrics. Filter for D-Brain, sigma=0.1, k_input=16. Compare rows with `dimensionality=2d` vs `dimensionality` empty (3D) for both `drcnet_hybrid_rgs` and `restormer_hybrid_rgs`. |
| `tmp/paper_final_k16_out/paper_tables/paper_runtime_summary.csv` | Computational cost comparison: 2D should be faster than 3D (fewer operations per slice, but more forward passes). |
| `src/plan_para_escribir_el_paper.md` | Overall paper plan (in Spanish). Section 1.4 (3D vs 2D critical ablation) explains why this experiment is mandatory for publication. |
| `src/J_invariance_DWMRI_denoising_report_hybrid_rgs.md` | Technical report. Describes the 3D architecture design and justification. Use for context on the architectural choices. |

## Prompt

Write the **3D vs 2D Convolutions** subsection for the Ablation Studies section of the paper. This is a **critical ablation** that will be scrutinized by reviewers — the narrative must be clear and defensible.

### Structure to produce

1. **Motivation** (1 paragraph, ~120-150 words)
   - Most prior self-supervised denoising methods for DWMRI use Conv2D, processing axial slices independently (e.g., MD-S2S, original Noise2Void adaptations).
   - Our method uses **Conv3D** to capture inter-slice dependencies (volumetric context).
   - Critical question: Is the performance improvement due to (A) the Hybrid RGS training scheme (angular self-supervision + masked reconstruction), or (B) the use of 3D convolutions, or (C) both?
   - This ablation isolates the architectural contribution by comparing **3D vs 2D with the same training scheme**.

2. **Experimental setup** (1 paragraph, ~100-120 words)
   - **3D baseline**: The proposed method as described (Conv3D/Transformer3D, K=16 RGS, Bernoulli masking on target channel).
   - **2D variant**: Replace Conv3D with Conv2D. Process each z-slice independently with K-channel input (same K=16). Apply the same RGS sampling and masked loss per slice.
   - Same hyperparameters: learning rate, epochs, mask_p=0.3, D-Brain dataset (σ=0.1).
   - Both architectures evaluated: DRCNet (naturally has a 2D variant) and Restormer (convert 3D attention to 2D per-slice attention).

3. **Results** (2 paragraphs, ~250-300 words)
   
   **Quantitative comparison:**
   - Present PSNR-ROI, SSIM-ROI, FA-MAE, and MD-MAE for 2D vs 3D for both architectures in a table.
   - **3D clearly outperforms 2D**: DRCNet-3D achieves +X dB over DRCNet-2D; Restormer-3D achieves +Y dB over Restormer-2D.
   - **DTI metric improvements**: FA-error and MD-error are Z% lower in 3D variants. This is crucial — it shows that volumetric context helps preserve anatomical coherence in downstream diffusion tensor estimation.
   - If 2D still beats classical baselines (e.g., better than MP-PCA), mention it: "Even the 2D variant outperforms classical baselines, validating the Hybrid RGS training scheme, but the 3D architecture provides substantial additional gains."

   **Interpretation:**
   - The 3D advantage comes from **inter-slice context**: axial slices in DWI are not independent — anatomical structures span multiple slices, and noise correlations exist across z.
   - 2D convolutions cannot exploit this volumetric coherence. Each slice is denoised independently, leading to potential slice-to-slice inconsistencies (visible in 3D renderings or streamlines).
   - **Conclusion**: Both the training scheme and the 3D architecture contribute to performance. The Hybrid RGS scheme enables effective self-supervised learning, and 3D convolutions ensure anatomical consistency.

4. **Computational cost note** (1 short paragraph, ~80-100 words)
   - 2D is faster per forward pass (fewer FLOPs per slice), but the gap is smaller than expected because 3D convolutions amortize computation across slices.
   - Training time: 2D is ~X% faster than 3D. Inference time: comparable or slightly faster for 2D.
   - However, the quality improvement of 3D justifies the modest computational overhead for clinical applications where accuracy is paramount.

5. **Table**: 3D vs 2D ablation
   - Rows: DRCNet-2D, DRCNet-3D, Restormer-2D, Restormer-3D.
   - Columns: Architecture, Dim, PSNR-ROI↑, SSIM-ROI↑, FA-MAE↓, MD-MAE↓, Params (M), Time/vol (s).
   - Bold the 3D results.

### Formatting requirements

- LaTeX source (compilable as a subsection).
- Use `\subsection{3D vs 2D Convolutions}`.
- Use `booktabs` for the table.
- Reference as `Table~\ref{tab:3d_vs_2d}`.
- Total target length: **1.5 pages** including table (approximately 550-700 words of body text).
- Do not include `\begin{document}` or preamble — just the subsection content.

### What to avoid

- Do not downplay the 2D results if they're still competitive with baselines. Frame it as: "2D validates the scheme, 3D adds volumetric consistency."
- Do not oversell 3D as a silver bullet. Be honest about the computational cost (even if modest).
- Do not claim 3D is "always better" for all tasks. Frame it as: "For DWMRI denoising, where volumetric coherence is critical for tractography and tensor estimation, 3D is essential."
- Do not compare against MD-S2S here (it's already in the main table). This ablation is **controlled**: same training scheme, only Conv2D vs Conv3D differs.

## Expected output

A single LaTeX code block containing:
- `\subsection{...}` with the full narrative text (~550-700 words).
- One table (3D vs 2D comparison for both architectures).
- Appropriate `\label`, `\cite`, and `\ref` commands.
- No preamble, no `\begin{document}`.
