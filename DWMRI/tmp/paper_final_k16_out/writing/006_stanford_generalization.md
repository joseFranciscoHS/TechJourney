# 006 — Generalization to Real Scanner Noise: Stanford HARDI

## Context for the LLM

You are helping write a research paper on **self-supervised denoising of Diffusion-Weighted MRI (DWI)** using a method called **Hybrid RGS** (Random Gradient Subset). The core idea: the model receives K randomly sampled gradient volumes as input channels and reconstructs one target volume using a blind-spot (J-invariant) loss. Two backbone architectures are evaluated: **DRCNet** (gated 3D CNN) and **Restormer** (3D transformer).

This prompt asks you to draft the **Stanford HARDI generalization subsection**, which demonstrates that the method works on **real scanner data** with a different acquisition protocol (no ground truth available). This is critical for establishing clinical relevance — all prior results used synthetic Rician noise on D-Brain.

**Key decisions already made:**
- **Stanford HARDI**: Real scanner noise, b=1000 (vs D-Brain b=2500), G=150 directions (vs D-Brain G=60). No ground truth.
- Evaluation is **qualitative**: visual assessment of denoised volumes, and **downstream metrics** (FA/MD map quality, anatomical plausibility).
- K sweep also performed on Stanford to show the method scales to larger shells.
- Baselines: Patch2Self and MD-S2S (both self-supervised, applicable without GT). MP-PCA could be included if run.
- **Tone**: Acknowledge limitations (no GT = no PSNR), but emphasize that FA/MD maps show clear improvement and anatomical coherence.

## Files to attach

Attach these files when you send this prompt. Paths are relative to `DWMRI/`.

| File | Purpose |
|------|---------|
| `tmp/paper_final_k16_out/paper_tables/paper_metrics_summary.csv` | Metrics for Stanford. Note: `psnr`, `ssim`, `mse` are reference-free (noisy as "reference"), not meaningful. FA/MD stats (if computed) or empty. Main value: qualitative assessment. |
| `tmp/paper_final_k16_out/paper_tables/paper_runtime_summary.csv` | Inference time on Stanford (G=150 is larger than D-Brain G=60). Shows scalability. |
| `src/plan_para_escribir_el_paper.md` | Overall paper plan (in Spanish). Section 4.3 (Stanford HARDI) explains the protocol and evaluation strategy without GT. |
| `src/J_invariance_DWMRI_denoising_report_hybrid_rgs.md` | Technical report. Describes the Stanford dataset and how the same model architecture transfers across protocols. |

## Prompt

Write the **Generalization to Real Scanner Noise** subsection for the Experimental Results or Generalization section of the paper. This demonstrates that the method works on **real-world data** with a different acquisition protocol.

### Structure to produce

1. **Motivation** (1 paragraph, ~120-150 words)
   - D-Brain results (previous sections) used synthetic Rician noise with known ground truth. This controlled setup enables quantitative metrics (PSNR, SSIM), but doesn't capture real scanner noise characteristics.
   - **Real scanner noise** includes: thermal noise, structured artifacts (Gibbs ringing, motion, eddy currents), hardware-specific patterns, and non-Rician distributions in low-SNR regions.
   - Stanford HARDI dataset: 150 gradient directions, b=1000 s/mm², real clinical scanner. No ground truth available (single acquisition).
   - Question: Does the Hybrid RGS method, trained on real noisy data without GT, produce anatomically plausible and smooth FA/MD maps? Does it scale to larger shells (G=150 vs 60)?

2. **Experimental setup** (1 paragraph, ~100-120 words)
   - **Stanford HARDI**: b=1000, G=150, 10 b0 volumes. Real Siemens scanner noise.
   - Training: Self-supervised on the Stanford data itself (K=16 RGS, same mask_p, same loss). No GT used.
   - K sweep: Evaluate K ∈ {5, 10, 16, 24, 30} to test scalability to large shells.
   - Baselines: Patch2Self (DIPY) and MD-S2S (both self-supervised, applicable without GT).
   - **Evaluation strategy**: Since no GT exists, assess via (1) visual inspection of denoised volumes, (2) FA/MD map smoothness and anatomical plausibility, (3) reference-free metrics (variance within homogeneous ROIs).

3. **Qualitative results** (2 paragraphs, ~250-300 words)
   
   **Visual quality:**
   - Denoised volumes show **clear noise reduction** while preserving edge sharpness (e.g., gray matter / white matter boundaries, ventricles).
   - Comparison with raw noisy volumes: proposed method removes granular noise without over-smoothing (no loss of fine anatomical detail).
   - Comparison with baselines: Patch2Self achieves moderate smoothing but retains residual noise in low-SNR regions (CSF, peripheral cortex). MD-S2S (Conv2D) shows slice-to-slice inconsistencies (flickering in 3D renderings). **Proposed method (3D)** produces volumetrically coherent denoising.

   **DTI-derived maps (FA/MD):**
   - **FA maps**: The proposed method produces smoother FA in white matter tracts (corpus callosum, corona radiata) without artificial over-smoothing that would inflate FA artificially.
   - **MD maps**: Anatomically plausible MD values in gray matter (~0.8 × 10⁻³ mm²/s) and white matter (~0.7 × 10⁻³ mm²/s). Baselines show noisier MD in CSF and gray matter boundaries.
   - **Qualitative scoring** (if available): Radiologist assessment or visual scoring on a 1-5 scale (sharpness, artifact presence, anatomical fidelity). Proposed method scores highest.
   - **No GT, but**: The fact that FA/MD maps are anatomically coherent and match known neuroanatomy is strong evidence of successful denoising without hallucination.

4. **K sweep on Stanford** (1 paragraph, ~120-150 words)
   - With G=150 (much larger than D-Brain G=60), test whether larger K is beneficial.
   - Results: K=16 performs well, K=24 shows marginal improvement (~0.2-0.5 dB in reference-free PSNR). K=30 plateaus.
   - **Scalability**: The method handles large shells without architectural changes. Inference time grows linearly with K (cite runtime table), but K=16 remains practical even for G=150.
   - Recommendation: K=16 is sufficient for G=150. For future ultra-high angular resolution protocols (G > 200), K=24 may be explored, but diminishing returns expected.

5. **Comparison with FiLM conditioning on Stanford** (1 short paragraph, ~80-100 words, optional)
   - If FiLM results on Stanford are available (from prompt 001), briefly mention: FiLM-conditioned models train successfully on Stanford. FA/MD maps show comparable or slightly improved anatomical coherence.
   - This demonstrates that the FiLM ablation (prompt 001) also generalizes to different protocols, not just D-Brain.

6. **Figure**: Stanford FA maps comparison
   - Layout: 1 row × N columns (Noisy, MP-PCA (if available), Patch2Self, MD-S2S, DRCNet-RGS, Restormer-RGS).
   - Show one representative axial slice (e.g., slice crossing corpus callosum).
   - Colormap: jet or hot, range [0, 1] for FA.
   - Caption: "FA maps on Stanford HARDI (b=1000, G=150). Proposed method (DRCNet-RGS and Restormer-RGS) produces smoother, anatomically coherent FA without over-smoothing. No ground truth available; assessment is qualitative."

7. **Optional table**: Stanford K sweep (reference-free metrics)
   - Rows: K ∈ {5, 10, 16, 24, 30} for DRCNet and Restormer.
   - Columns: K, Ref-free PSNR (noisy as ref), FA variance in WM ROI↓, MD variance in WM ROI↓, Time/vol (s).
   - Lower variance = smoother, more stable denoising.

### Formatting requirements

- LaTeX source (compilable as a subsection or section).
- Use `\subsection{Generalization to Real Scanner Noise}` or `\section{Evaluation on Real Clinical Data}`.
- Reference the figure as `Figure~\ref{fig:stanford_fa_comparison}`.
- If including a table, use `booktabs` and reference as `Table~\ref{tab:stanford_k_sweep}`.
- Total target length: **2-2.5 pages** including figure (approximately 650-850 words of body text).
- Do not include `\begin{document}` or preamble — just the section content.

### What to avoid

- Do not claim quantitative improvements without GT. Frame results as "qualitative assessment" and "anatomically plausible."
- Do not hide the limitation that Stanford has no GT. Acknowledge it upfront, then pivot to downstream metrics (FA/MD) as the evaluation proxy.
- Do not compare PSNR/SSIM values from Stanford to D-Brain (different acquisition protocols, incomparable).
- Do not oversell visual results. Use measured language: "clear improvement," "anatomically coherent," "reduced noise," rather than "perfect" or "artifact-free."

## Expected output

A single LaTeX code block containing:
- `\subsection{...}` or `\section{...}` with the full narrative text (~650-850 words).
- One figure reference (Stanford FA maps comparison).
- Optionally, one table (K sweep on Stanford).
- Appropriate `\label`, `\cite`, and `\ref` commands.
- No preamble, no `\begin{document}`.
