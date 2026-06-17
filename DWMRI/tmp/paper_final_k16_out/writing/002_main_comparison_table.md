# 002 — Main Comparison Table and Results

## Context for the LLM

You are helping write a research paper on **self-supervised denoising of Diffusion-Weighted MRI (DWI)** using a method called **Hybrid RGS** (Random Gradient Subset). The core idea: the model receives K randomly sampled gradient volumes as input channels and reconstructs one target volume using a blind-spot (J-invariant) loss. Two backbone architectures are evaluated: **DRCNet** (gated 3D CNN) and **Restormer** (3D transformer).

This prompt asks you to draft the **main experimental results section**, including the primary comparison table that positions the proposed method against classical baselines and a supervised upper bound. This is the **central section of the paper** — all other sections (ablations, robustness studies) build on this foundation.

**Key decisions already made:**
- The main comparison uses **K=16** for both architectures (optimal K from ablation studies).
- Dataset: **D-Brain with Rician noise σ=0.1** (standard noise level).
- Baselines included: MP-PCA, Patch2Self (DIPY backend), MD-S2S (Conv2D).
- Upper bound: supervised versions of both DRCNet and Restormer with GT access.
- The narrative should emphasize that the self-supervised method **closes the gap** with classical baselines while remaining practical for real clinical data (no GT required).

## Files to attach

Attach these files when you send this prompt. Paths are relative to `DWMRI/`.

| File | Purpose |
|------|---------|
| `tmp/paper_final_k16_out/paper_tables/paper_metrics_summary.csv` | Complete metrics for all methods: PSNR, SSIM, MSE (full and ROI), FA/MD/AD/RD errors. Filter rows where `dataset=dbrain`, `sigma=0.1`, and `k_input=16` (or empty for baselines). |
| `tmp/paper_final_k16_out/paper_tables/paper_runtime_summary.csv` | Training time, inference time per volume, total time, and parameter counts for all methods. |
| `src/plan_para_escribir_el_paper.md` | Overall paper plan (in Spanish). Sections 2 (baselines) and 6.4 (main comparison table structure) describe what to include and how to frame the comparison. |
| `src/J_invariance_DWMRI_denoising_report_hybrid_rgs.md` | Technical report with method details. Use for background on why certain design choices were made. |

## Prompt

Write the **Experimental Results** section for the main body of the paper. This section presents the primary quantitative comparison between the proposed method and all baselines.

### Structure to produce

1. **Introduction paragraph** (~100-150 words)
   - Brief reminder of the experimental setup: D-Brain dataset, Rician noise σ=0.1, K=16 for Hybrid RGS.
   - Preview of what this section shows: the proposed method outperforms classical baselines on both image quality metrics (PSNR/SSIM) and downstream DTI metrics (FA/MD errors).
   - Mention the supervised upper bound as context for the self-supervised gap.

2. **Classical baselines discussion** (~150-200 words)
   - **MP-PCA**: Standard preprocessing baseline. Achieves X dB PSNR-ROI, but limited by linear PCA assumptions.
   - **Patch2Self (DIPY)**: Self-supervised angular baseline using OLS regression. Achieves Y dB PSNR-ROI. Our method improves on this by using 3D convolutions instead of linear regression.
   - **MD-S2S**: Self-supervised with Conv2D and Bernoulli masking. Achieves Z dB PSNR-ROI, but processes slices independently (no inter-slice context).
   - Frame the comparison: the proposed method advances beyond these baselines by combining angular self-supervision (like P2S), spatial masking (like MD-S2S), and 3D context.

3. **Proposed method results** (~200-250 words)
   - **DRCNet-Hybrid-RGS**: Achieves A dB PSNR-ROI, B FA-MAE, C MD-MAE. Highlight improvements over baselines (e.g., +X% over MP-PCA in MSE-ROI, -Y% in FA-error over Patch2Self).
   - **Restormer-Hybrid-RGS**: Achieves D dB PSNR-ROI (slightly better than DRCNet), E FA-MAE, F MD-MAE. Note architecture differences: Restormer has richer representational capacity but higher computational cost.
   - Compare the two architectures: DRCNet is faster and lighter, Restormer has slightly better PSNR but comparable DTI metrics.
   - Emphasize **DTI metric preservation**: both models achieve FA/MD errors competitive with or better than baselines, showing that the self-supervised training does not introduce downstream bias.

4. **Supervised upper bound analysis** (~150-200 words)
   - Supervised DRCNet and Restormer represent the ceiling with GT access. Report their PSNR-ROI, FA-error, MD-error.
   - Calculate the **self-supervised gap**: how many dB of PSNR the self-supervised version loses compared to supervised.
   - If the gap is small (< 2-3 dB), emphasize this as a strength: "the self-supervised method achieves XX% of the supervised performance while requiring no ground truth, making it practical for clinical deployment."
   - Discuss why the supervised models don't perfectly denoise: Rician noise is signal-dependent, and even with GT, some residual noise remains in low-SNR regions.

5. **Main comparison table** (LaTeX `booktabs`)
   - Rows: Noisy (no denoising), MP-PCA, Patch2Self, MD-S2S, DRCNet-RGS, Restormer-RGS, DRCNet-Supervised, Restormer-Supervised
   - Columns: Method, PSNR↑, SSIM↑, MSE↓, PSNR-ROI↑, MSE-ROI↓, FA-MAE↓, MD-MAE↓, Time/vol (s)
   - Bold the best self-supervised result per column. Italicize the supervised upper bound.
   - Include a caption explaining: "All methods evaluated on D-Brain with Rician noise σ=0.1. Self-supervised methods use K=16 gradient volumes. ROI metrics computed over brain tissue only (threshold > 0.02). Time/vol is inference time per gradient volume."

### Formatting requirements

- LaTeX source (compilable as a section within a larger document).
- Use `\section{Experimental Results}` or `\subsection{Main Comparison}` depending on where this fits in your paper structure.
- Use `booktabs` for the table (`\toprule`, `\midrule`, `\bottomrule`).
- Cite baselines appropriately: MP-PCA `\cite{veraart2016mppca}`, Patch2Self `\cite{fadnavis2020patch2self}`, MD-S2S `\cite{kang2021mds2s}`.
- Reference the table as `Table~\ref{tab:main_comparison}`.
- Total target length: **2-3 pages** including table (approximately 700-1000 words of body text).
- Do not include `\begin{document}` or preamble — just the section content.

### What to avoid

- Do not oversell the results. Frame improvements as "consistent" or "substantial" rather than "dramatic" unless the difference is > 5 dB.
- Do not hide weaknesses. If MD-S2S performs surprisingly well in some metric, acknowledge it and discuss why.
- Do not present the supervised upper bound as a failure. It's a realistic ceiling given the noise model.
- Do not include Stanford results here (they go in a separate generalization section; no GT for quantitative comparison).

## Expected output

A single LaTeX code block containing:
- `\section{...}` or `\subsection{...}` with the full narrative text (~700-1000 words).
- One `table` environment with the main comparison results (all methods, all key metrics).
- Appropriate `\label`, `\cite`, and `\ref` commands.
- No preamble, no `\begin{document}`.
