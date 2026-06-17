# 005 — Robustness to Noise Level: Sigma Sweep

## Context for the LLM

You are helping write a research paper on **self-supervised denoising of Diffusion-Weighted MRI (DWI)** using a method called **Hybrid RGS** (Random Gradient Subset). The core idea: the model receives K randomly sampled gradient volumes as input channels and reconstructs one target volume using a blind-spot (J-invariant) loss. Two backbone architectures are evaluated: **DRCNet** (gated 3D CNN) and **Restormer** (3D transformer).

This prompt asks you to draft the **robustness to noise level subsection**, which demonstrates that the proposed method performs consistently across different signal-to-noise regimes. This addresses a common reviewer concern: "Does the method only work at one specific noise level, or is it robust?"

**Key decisions already made:**
- Evaluate on **Rician noise σ ∈ {0.05, 0.10, 0.15, 0.20}** on D-Brain (synthetic noise with GT).
- For each σ, both training and inference use that noise level (not cross-noise-level transfer).
- Compare proposed method against baselines (MP-PCA, Patch2Self, MD-S2S) at each σ.
- Results show **consistent advantage** across all noise levels (no catastrophic failure at high or low SNR).

## Files to attach

Attach these files when you send this prompt. Paths are relative to `DWMRI/`.

| File | Purpose |
|------|---------|
| `tmp/paper_final_k16_out/paper_tables/paper_metrics_summary.csv` | Complete metrics. Filter for D-Brain with `noise_rician_sigma` ∈ {0.05, 0.10, 0.15, 0.20}. Extract PSNR-ROI, FA-MAE for all methods (baselines + proposed architectures) at each sigma. |
| `src/plan_para_escribir_el_paper.md` | Overall paper plan (in Spanish). Section 4.2 (noise level variation) explains the protocol and expected outcome. |
| `src/J_invariance_DWMRI_denoising_report_hybrid_rgs.md` | Technical report. Describes the Rician noise model and how it's applied during training. |

## Prompt

Write the **Robustness to Noise Level** subsection for the Ablation Studies or Robustness Analysis section of the paper. This demonstrates that the method generalizes across different SNR regimes.

### Structure to produce

1. **Motivation** (1 paragraph, ~100-120 words)
   - Real-world DWMRI scans have varying noise levels depending on: scanner hardware, acquisition time, field strength (1.5T vs 3T vs 7T), and patient motion.
   - A practical denoising method must work across a range of SNR conditions, not just at a single "sweet spot."
   - Question: Does the Hybrid RGS method maintain its advantage over baselines at low noise (σ=0.05, high SNR) and high noise (σ=0.20, low SNR), or does performance degrade asymmetrically?

2. **Experimental setup** (1 paragraph, ~80-100 words)
   - Evaluate σ ∈ {0.05, 0.10, 0.15, 0.20} on D-Brain.
   - For each σ, train models from scratch with that noise level (not one model for all sigmas).
   - Baselines (MP-PCA, Patch2Self, MD-S2S) also evaluated at each σ.
   - Metrics: PSNR-ROI (image quality) and FA-MAE (downstream diffusion metric preservation).
   - All other settings fixed: K=16, same hyperparameters.

3. **Results and analysis** (2-3 paragraphs, ~300-350 words)
   
   **Overall trend:**
   - All methods show decreasing PSNR as σ increases (expected — more noise is harder to remove).
   - The proposed method (both DRCNet and Restormer) **maintains a consistent advantage** over baselines across all σ values.
   - Gap over MP-PCA: ~X dB at σ=0.05, ~Y dB at σ=0.20. The gap is stable or slightly increases at high noise (hypothesis: learned methods adapt better to severe noise than linear PCA).

   **Low noise regime (σ=0.05):**
   - High SNR scenario. Denoising is "easier" — all methods achieve PSNR > 29 dB.
   - Proposed method achieves A dB PSNR-ROI, outperforming MP-PCA by B dB and P2S by C dB.
   - At low noise, even simple methods work reasonably well, so the gap is smaller. However, FA-error differences remain significant: proposed method achieves D% lower FA-error than baselines.

   **High noise regime (σ=0.20):**
   - Challenging low-SNR scenario. Baselines struggle: MP-PCA achieves only E dB PSNR-ROI.
   - Proposed method achieves F dB PSNR-ROI, a G dB improvement over MP-PCA.
   - **DTI preservation under high noise**: FA-error remains controlled at H (vs I for MP-PCA). This is critical — the method doesn't "hallucinate" structure in low-SNR regions.

   **Architecture comparison:**
   - DRCNet and Restormer show similar robustness trends. Restormer has a slight edge at σ=0.05 (high SNR), DRCNet is more stable at σ=0.20 (possible overfitting of Restormer at high noise due to larger capacity).

   **Conclusion:**
   - The method is **robust across SNR regimes**. No catastrophic failure at high noise, no diminishing returns at low noise.
   - Recommendation: σ=0.10 as the standard benchmark (middle of the range), but users can apply the same model architecture across protocols with varying noise.

4. **Figure**: PSNR-ROI vs σ curves
   - X-axis: σ ∈ {0.05, 0.10, 0.15, 0.20}.
   - Y-axis: PSNR-ROI (dB).
   - Curves: Noisy (no denoising), MP-PCA, Patch2Self, MD-S2S, DRCNet-RGS, Restormer-RGS.
   - Proposed methods should be clearly on top across all σ.

5. **Table**: Robustness summary (optional, can be in supplementary)
   - Rows: σ values {0.05, 0.10, 0.15, 0.20}.
   - Columns: Noisy PSNR, MP-PCA, P2S, MD-S2S, DRCNet-RGS, Restormer-RGS (all PSNR-ROI).
   - Keep concise — full metrics in supplementary material.

### Formatting requirements

- LaTeX source (compilable as a subsection).
- Use `\subsection{Robustness to Noise Level}`.
- Reference the figure as `Figure~\ref{fig:sigma_robustness}`.
- If including a table, use `booktabs` and reference as `Table~\ref{tab:sigma_sweep}`.
- Total target length: **1.5-2 pages** including figure (approximately 500-650 words of body text).
- Do not include `\begin{document}` or preamble — just the subsection content.

### What to avoid

- Do not claim "perfect robustness" — performance does degrade at high noise (as expected). Frame it as "consistent relative advantage."
- Do not hide baseline performance at low noise. If MP-PCA is close at σ=0.05, acknowledge it and emphasize that the gap remains in DTI metrics.
- Do not compare cross-noise transfer (e.g., train at σ=0.10, test at σ=0.20). That's a separate experiment. This subsection is: train and test at the same σ, vary σ.

## Expected output

A single LaTeX code block containing:
- `\subsection{...}` with the full narrative text (~500-650 words).
- One figure reference (sigma robustness curve).
- Optionally, one table (concise summary).
- Appropriate `\label`, `\cite`, and `\ref` commands.
- No preamble, no `\begin{document}`.
