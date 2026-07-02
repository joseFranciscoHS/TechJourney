# 001 — Gradient Direction Conditioning via FiLM

## Context for the LLM

You are helping write a research paper on **self-supervised denoising of Diffusion-Weighted MRI (DWI)** using a method called **Hybrid RGS** (Random Gradient Subset). The core idea: the model receives K randomly sampled gradient volumes as input channels and reconstructs one target volume using a blind-spot (J-invariant) loss. Two backbone architectures are evaluated: **DRCNet** (gated 3D CNN) and **Restormer** (3D transformer).

This prompt asks you to draft the **gradient direction conditioning ablation subsection**. It belongs in the Ablation Studies section of the paper, after the core methodology and main comparison table have been presented.

**Key decisions already made:**
- An initial naive approach (additive input encoding) was explored and **discarded for theoretical reasons** before producing valid experimental results. It should be mentioned briefly as motivation, not as a failed experiment.
- **FiLM conditioning** (Feature-wise Linear Modulation) replaced it and has full experimental results.
- The comparison is **Baseline RGS vs FiLM** only (two-way, not three-way).
- The tone should frame FiLM as a "natural extension that tests whether explicit conditioning helps" rather than a critical missing component.

## Files to attach

Attach these files when you send this prompt. Paths are relative to `DWMRI/`.

| File | Purpose |
|------|---------|
| `src/J_invariance_DWMRI_denoising_report_hybrid_rgs.md` | Technical report with full descriptions of the motivation (§4.4), FiLM design (§4.5), data pipeline, and inference details. This is your primary technical reference. |
| `tmp/paper_final_k16_out/orientation_conditioning_metrics_summary.md` | Complete metrics tables: Baseline vs FiLM for D-Brain (DRCNet + Restormer) and Stanford FiLM-only results. Contains PSNR, SSIM, MSE, FA/MD/AD/RD errors, parameter counts, and timing. |
| `tmp/paper_final_k16_out/paper_section_structure_orientation_conditioning.md` | Proposed section structure, narrative flow, table/figure specifications, discussion points, and writing style recommendations. Follow this structure closely. |
| `src/plan_para_escribir_el_paper.md` | Overall paper plan (in Spanish). Section 1.8 describes the FiLM ablation design, hypotheses, and expected outputs. Use this for context on where the ablation fits in the paper. |
| `src/utils/film_layer.py` | FiLM implementation (59 lines). Reference for precise technical description. |
| `src/utils/orientation_encoder.py` | Discarded additive encoder (53 lines). Reference for the brief "naive approach" paragraph. |

## Prompt

Write the **Gradient Direction Conditioning via FiLM** subsection for the ablation studies section of the paper. Follow the structure and recommendations in `paper_section_structure_orientation_conditioning.md` closely.

### Structure to produce

1. **Motivation** (1 paragraph, ~100-150 words)
   - In RGS, the K-channel stack is randomly shuffled each step; the network has no signal about which physical gradient direction corresponds to the target.
   - Hypothesis: explicit direction awareness could help in anatomically complex regions (crossing fibers).
   - Analogy with positional encodings.
   - Keep tone measured: "natural question to explore," not "critical gap."

2. **Discarded naive approach** (1 short paragraph, ~80-120 words)
   - Briefly describe the additive input encoding: project `[cos_x, cos_y, cos_z, b_norm]` → learned 2D spatial pattern → interpolate → add to all K input channels before the backbone.
   - Why it was discarded: generates artificial spatial patterns from spatially invariant metadata; signal dilution through layers; overconditions all K channels instead of only the target.
   - Do NOT present this as an experiment that failed — frame it as a design iteration discarded on theoretical grounds.

3. **FiLM Conditioning** (2-3 paragraphs, ~200-300 words)
   - Core mechanism: $\text{FiLM}(F) = \gamma \odot F + \beta$ where $\gamma, \beta$ are predicted from the 4D condition vector of the **target channel only** via a small MLP.
   - Five key design differences from the additive approach: (i) target-only conditioning, (ii) feature-space modulation, (iii) multiple injection points, (iv) no artificial spatial patterns, (v) identity initialization.
   - Implementation details: placement in DRCNet (2 FiLM layers) and Restormer (3 FiLM layers); parameter overhead (~4% in both); preserves J-invariance because the condition is deterministic metadata.

4. **Experimental Results** (2 paragraphs, ~200-250 words)
   - **D-Brain** quantitative results in a table (see metrics summary).
   - **Full-volume improvement**: +1.47 dB PSNR and -28.8% MSE for DRCNet; +0.77 dB PSNR and -16.2% MSE for Restormer.
   - **ROI metrics show degradation**: PSNR-ROI decreases by -0.64 dB (DRCNet) and -0.53 dB (Restormer); ROI MSE increases by +15.8% (DRCNet) and +12.9% (Restormer).
   - **Tensor metrics**: FA-MAE improves in DRCNet (-7.0%) but worsens in Restormer (+9.5%); MD-MAE changes little in DRCNet and improves slightly in Restormer.
   - **Interpretation**: FiLM may improve global smoothness while degrading fine anatomical detail in ROI, or identity initialization may need tuning.
   - **Stanford** generalization: FiLM trains successfully on a different acquisition protocol (G=150, b=2000). No baseline comparison available; qualitative FA/MD assessment only. Stanford PSNR is noisy-vs-noisy, not restoration accuracy.
   - Architecture dependency: Results suggest FiLM requires architecture-specific tuning; DRCNet gains more full-volume PSNR than Restormer.

5. **Table** (LaTeX `booktabs`)
   - Baseline vs FiLM for both architectures on D-Brain.
   - Columns: Model, Conditioning, PSNR (dB), PSNR-ROI (dB), FA-MAE, MD-MAE, Params, Time/vol.
   - Include a delta row per architecture.

### Formatting requirements

- LaTeX source (compilable as a subsection within a larger document).
- Use `\subsection{Gradient Direction Conditioning via FiLM}`.
- Use `booktabs` for the table (`\toprule`, `\midrule`, `\bottomrule`).
- Cite FiLM as `\cite{perez2018film}` (Perez et al., 2018, AAAI).
- Reference the FiLM table as `Table~\ref{tab:film_ablation}`.
- Total target length: **1–1.5 pages** of text + 1 table (approximately 600-900 words of body text).
- Do not include `\begin{document}` or preamble — just the subsection content.

### What to avoid

- Do not oversell FiLM as a breakthrough; it shows mixed results (full-volume gain, ROI degradation).
- Do not hide the Restormer FA degradation — discuss it honestly as architecture-specific behavior.
- Do not describe the discarded additive approach as a full experiment.
- Do not include Stanford metrics in the quantitative table (no GT, no baseline for comparison).

## Expected output

A single LaTeX code block containing:
- `\subsection{...}` with the full narrative text (~600-900 words).
- One `table` environment with the FiLM ablation results.
- Appropriate `\label` and `\cite` commands.
- No preamble, no `\begin{document}`.
