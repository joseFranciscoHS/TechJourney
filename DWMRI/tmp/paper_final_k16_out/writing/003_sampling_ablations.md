# 003 — Sampling Ablations: Sequential vs RGS and K Sweep

## Context for the LLM

You are helping write a research paper on **self-supervised denoising of Diffusion-Weighted MRI (DWI)** using a method called **Hybrid RGS** (Random Gradient Subset). The core idea: the model receives K randomly sampled gradient volumes as input channels and reconstructs one target volume using a blind-spot (J-invariant) loss. Two backbone architectures are evaluated: **DRCNet** (gated 3D CNN) and **Restormer** (3D transformer).

This prompt asks you to draft the **sampling ablation subsection**, which validates two core design choices: (1) **random gradient sampling (RGS) vs sequential windowing**, and (2) **the optimal number of input volumes K**. These ablations belong in the Ablation Studies section, after the main comparison table.

**Key decisions already made:**
- These are **critical ablations** that justify the method's name (Random Gradient Subset) and demonstrate the value of angular diversity during training.
- The K sweep establishes **K=16 as the recommended configuration** (good quality/cost tradeoff).
- Both architectures (DRCNet and Restormer) are evaluated to show robustness.
- Results are on **D-Brain only** (controlled conditions with GT).

## Files to attach

Attach these files when you send this prompt. Paths are relative to `DWMRI/`.

| File | Purpose |
|------|---------|
| `tmp/paper_final_k16_out/paper_tables/paper_metrics_summary.csv` | Complete metrics. Filter for D-Brain, sigma=0.1. For Sequential vs RGS: compare rows with `sampling_mode=sequential` vs `sampling_mode=rgs` at `k_input=16`. For K sweep: compare rows with `k_input={5,10,16,24,30}` for both architectures. |
| `tmp/paper_final_k16_out/paper_tables/paper_runtime_summary.csv` | Training time and inference time as a function of K. Essential for discussing the quality/cost tradeoff. |
| `src/plan_para_escribir_el_paper.md` | Overall paper plan (in Spanish). Section 1.1 (Sequential vs RGS) and 1.2 (K sweep) describe the experimental protocol and expected outcomes. |
| `src/J_invariance_DWMRI_denoising_report_hybrid_rgs.md` | Technical report. Section on RGS sampling explains the implementation details and why angular diversity acts as implicit data augmentation. |

## Prompt

Write the **Sampling Ablations** subsection for the Ablation Studies section of the paper. This subsection groups two related experiments that validate the sampling strategy.

### Structure to produce

#### Part 1: Sequential vs Random Gradient Sampling (~300-400 words)

1. **Motivation** (1 paragraph, ~100 words)
   - In RGS, each training step samples K distinct gradient indices randomly from the full shell G. In sequential mode, the network processes sliding windows of K consecutive gradients deterministically.
   - Question: Does the random shuffling provide value (implicit data augmentation through angular diversity), or is it equivalent to systematic windowing with the same K?
   - This tests whether **diversity during training** matters beyond just the number of input channels.

2. **Experimental setup** (1 paragraph, ~80-100 words)
   - Both modes use K=16, same hyperparameters (learning rate, epochs, mask_p), same D-Brain dataset.
   - Sequential mode: sliding windows [0:16], [1:17], ..., [G-K:G] with the last channel as target.
   - RGS mode: each batch samples K distinct indices uniformly without replacement, target always at channel 15.
   - Both architectures evaluated.

3. **Results** (1 paragraph, ~120-150 words)
   - Present PSNR-ROI and FA-error for both modes and both architectures in a small comparison table.
   - Highlight that **RGS outperforms sequential** by X dB (DRCNet) and Y dB (Restormer) in PSNR-ROI.
   - Interpretation: Random sampling acts as angular data augmentation — the model learns to denoise across diverse angular contexts rather than memorizing local angular neighborhoods.
   - If sequential performs surprisingly close, discuss: the gap is smaller than expected, suggesting that the J-invariance principle itself (blind-spot masking) is the dominant factor, with RGS providing a modest but consistent boost.

#### Part 2: K Sweep — Number of Input Volumes (~400-500 words)

1. **Motivation** (1 paragraph, ~100 words)
   - How many gradient volumes K are necessary for effective denoising? Too few → insufficient angular information. Too many → diminishing returns, higher memory/compute.
   - Finding the optimal K establishes practical guidelines for different acquisition protocols (e.g., D-Brain G=60 vs Stanford G=150).

2. **Experimental setup** (1 paragraph, ~80-100 words)
   - Evaluate K ∈ {5, 10, 16, 24, 30} on D-Brain (σ=0.1).
   - Each K requires training a separate model (input channel dimension changes).
   - DRCNet used for full sweep, Restormer validated at key K values.
   - Same training protocol for all K.

3. **Results and analysis** (2 paragraphs, ~220-280 words)
   - **Quality vs K curve**: PSNR-ROI increases rapidly from K=5 to K=16, then plateaus. K=24 and K=30 show marginal improvements (< 0.5 dB over K=16).
   - **Compute vs K tradeoff**: Training time and inference time grow approximately linearly with K (cite runtime table). K=16 offers the best balance: 95%+ of the quality of K=30 at 50% of the cost.
   - **Architecture differences**: DRCNet shows slightly larger gains from increasing K than Restormer (hypothesis: transformer's self-attention partially compensates for smaller K by better exploiting the available angular information).
   - **Recommended configuration**: K=16 for most DWMRI protocols with G ≥ 60. For very large shells (G > 100), K=24 may be beneficial if compute budget allows.

4. **Figure**: PSNR-ROI vs K curve
   - Two curves: DRCNet and Restormer.
   - X-axis: K ∈ {5, 10, 16, 24, 30}.
   - Y-axis: PSNR-ROI (dB).
   - Annotate the recommended K=16 point.

5. **Table**: Sampling ablations
   - Row groups: (A) Sequential vs RGS at K=16, (B) K sweep for both architectures.
   - Columns: Architecture, Sampling/K, PSNR-ROI↑, FA-MAE↓, Params (M), Time/vol (s).
   - Keep concise — detailed metrics in supplementary material.

### Formatting requirements

- LaTeX source (compilable as a subsection).
- Use `\subsection{Sampling Strategy Ablations}`.
- Use `booktabs` for the table.
- Reference as `Table~\ref{tab:sampling_ablations}` and `Figure~\ref{fig:k_sweep}`.
- Total target length: **2 pages** including table and figure (approximately 700-900 words of body text).
- Do not include `\begin{document}` or preamble — just the subsection content.

### What to avoid

- Do not present these as surprising results. Frame them as validation of design intuitions (random sampling is better, K=16 is optimal).
- Do not hide the plateau effect at high K. It's actually a positive result: the method doesn't require excessive K.
- Do not oversell the sequential vs RGS difference if it's < 1 dB. Call it "consistent" rather than "dramatic."

## Expected output

A single LaTeX code block containing:
- `\subsection{...}` with the full narrative text (~700-900 words).
- One table (sampling ablations summary).
- One figure reference (K sweep curve — the actual figure generation is separate).
- Appropriate `\label`, `\cite`, and `\ref` commands.
- No preamble, no `\begin{document}`.
