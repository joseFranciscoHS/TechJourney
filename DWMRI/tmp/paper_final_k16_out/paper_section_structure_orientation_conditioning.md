# Paper Section Structure: Gradient Direction Conditioning

## Proposed Structure for the Orientation Conditioning Section

This document proposes the structure, narrative flow, tables, and figures for the gradient direction conditioning section of the paper.

---

## Section Title and Placement

**Suggested title:** "Gradient Direction Conditioning via FiLM"

**Placement:** This should be a subsection within the Methods or an Ablation Studies section, positioned after the core RGS methodology is described but before generalization/robustness experiments.

---

## Narrative Flow

### 1. Motivation (1 paragraph, ~100-150 words)

**Key points:**
- In standard RGS, the network receives K randomly sampled gradients in random order at each training step
- The network has no explicit signal about **which** physical gradient direction corresponds to the target volume
- **Hypothesis:** Explicit gradient direction awareness could help the network in anatomically complex regions (e.g., crossing fibers) by allowing it to selectively modulate features aligned with that direction
- **Analogy:** Positional encodings in transformers inform the model about sequence position; orientation encoding informs the model about q-space position

**Tone:** Motivate the problem clearly but avoid overselling - frame as "natural question to explore" rather than "critical missing component"

---

### 2. Naive Approach and Its Limitations (1 paragraph, ~100-150 words)

**Key points:**
- **Initial exploration:** Additive input encoding - learned 2D pattern from gradient metadata added to input channels
- **Fundamental limitation:** Generates artificial **spatial patterns** from **spatially invariant** metadata (gradient direction is uniform across the field of view)
- Additional issues: signal dilution through layers, spatial asymmetry (broadcast along one axis), conditioning all K channels instead of only the target
- **Decision:** Approach discarded for theoretical reasons before extensive evaluation

**Tone:** Brief and factual - acknowledge the exploration but focus on why it was insufficient, not defensive about the decision

---

### 3. FiLM Conditioning Design (2-3 paragraphs, ~200-300 words)

#### 3a. Core mechanism (1 paragraph)
- FiLM (Feature-wise Linear Modulation): $\text{FiLM}(F) = \gamma \odot F + \beta$
- Conditioning vector: `[cos_x, cos_y, cos_z, b_norm]` for **target volume only**
- MLP generates $\gamma, \beta$ per channel (spatially broadcast)
- Identity initialization: network starts as no-op, learns how much conditioning to apply

#### 3b. Key design differences (1 paragraph, bulleted or enumerated)
1. **Target-only conditioning** (not all K channels)
2. **Feature-space modulation** (not input-space)
3. **Multiple injection points** in encoder-decoder hierarchy (not single input-level bias)
4. **No artificial spatial patterns** (scalars broadcast spatially)
5. **Adaptive learning** (identity init allows network to determine conditioning strength)

#### 3c. Implementation details (1 paragraph)
- Placement: DRCNet (2 layers: bottleneck, decoder), Restormer (3 layers: enc2, latent, dec2)
- Parameter overhead: ~4.5K params (+3.9% for DRCNet), ~6.8K params (+3.8% for Restormer)
- Preserves J-invariance: conditioning is deterministic metadata, not noisy observations
- Minimal inference overhead: ~4% time increase per volume

---

### 4. Experimental Results (2-3 paragraphs, ~250-350 words)

#### 4a. D-Brain quantitative comparison (1-2 paragraphs)
- **DRCNet improvements:** +1.47 dB PSNR (full), +1.92 dB PSNR-ROI, -38% MSE-ROI, -7.7% FA error
- **Restormer improvements:** +0.77 dB PSNR (full), +1.38 dB PSNR-ROI, -33.7% MSE-ROI, but +7.4% FA error (degradation)
- **Architecture dependency:** DRCNet (simpler CNN) benefits more than Restormer (richer transformer self-attention)
- **Clinical relevance:** ROI improvements more significant than full-volume metrics (38% vs 29% MSE reduction in DRCNet)

#### 4b. Stanford generalization (1 paragraph)
- Successfully applied to Stanford dataset (G=150, b=1000 vs D-Brain G=60, b=2500)
- No clean reference, so qualitative assessment via FA/MD maps
- (Optionally mention: no baseline Stanford run available for comparison, but method trains successfully on larger shell)

#### 4c. Restormer FA degradation discussion (optional, 1 paragraph)
- Hypothesize reasons: overfitting, suboptimal FiLM placement for transformer architecture, initialization sensitivity
- Future work: architecture-specific FiLM design, regularization, or different conditioning strategies for attention-based models

---

## Tables

### Table 1: Quantitative Comparison (D-Brain, sigma=0.1)

| Model | Conditioning | PSNR (dB)↑ | PSNR-ROI (dB)↑ | FA-MAE↓ | MD-MAE↓ | Params | Time/vol |
|-------|--------------|------------|----------------|---------|---------|--------|----------|
| DRCNet | None | 23.93 | 28.48 | 0.261 | 3539 | 116K | 34.0s |
| DRCNet | **FiLM** | **25.40** | **30.40** | **0.241** | **3452** | 120K | 35.3s |
| Δ DRCNet | | **+1.47** | **+1.92** | **-7.7%** | **-2.5%** | +3.9% | +3.8% |
| Restormer | None | 22.83 | 26.56 | **0.242** | 3502 | 178K | 126s |
| Restormer | **FiLM** | **23.60** | **27.94** | 0.260 | **3423** | 185K | 130s |
| Δ Restormer | | **+0.77** | **+1.38** | **+7.4%** | **-2.3%** | +3.8% | +3.2% |

**Caption:** *Comparison of baseline RGS and FiLM conditioning on D-Brain dataset (σ=0.1, K=16, G=60). Bold indicates best result per model family. FiLM improves PSNR and ROI metrics in both architectures, with larger gains in DRCNet. Restormer FA-MAE degrades with FiLM, suggesting architecture-specific tuning needs. Minimal computational overhead (3-4% inference time).*

---

## Figures

### Figure 1: FA/MD Map Comparison (D-Brain)

**Layout:** 3×2 grid (3 rows: Noisy, Baseline RGS, FiLM; 2 columns: FA map, MD map)

**Specific slices:** Choose an axial slice with crossing fibers or complex white matter (e.g., centrum semiovale, corona radiata intersection)

**Visual elements:**
- Color bar for FA (0-1) and MD (0-3000 μm²/s standard range)
- ROI overlay or zoom inset highlighting improvement region
- Difference maps (optional): |FiLM - GT| vs |Baseline - GT| to show error reduction

**Caption:** *Fractional Anisotropy (FA) and Mean Diffusivity (MD) maps on D-Brain test data. Top: Noisy input. Middle: Baseline RGS (no conditioning). Bottom: FiLM conditioning. FiLM reduces FA error by 7.7% and MD error by 2.5% in DRCNet, with visually smoother maps and better preservation of fiber structure (white box). Colorbar: FA [0,1], MD [0, 3000] μm²/s.*

### Figure 2 (Optional): Stanford Qualitative Comparison

**Layout:** 1×2 or 2×2 (FA and MD maps for FiLM Stanford results)

**Purpose:** Demonstrate generalization to different acquisition (b=1000, G=150)

**Caption:** *FiLM conditioning applied to Stanford dataset (b=1000, G=150, no ground truth). FA and MD maps show smooth, anatomically plausible structures. Generalization to larger shells and different b-values successful.*

---

## Discussion Points (for main Discussion section)

1. **Why FiLM works:** Feature-space modulation at intermediate layers is more effective than input-space patterns. The network can selectively amplify or suppress features relevant to each gradient direction.

2. **Architecture dependency:** Simpler architectures (DRCNet) benefit more from explicit conditioning than richer architectures (Restormer) with built-in relational reasoning (self-attention). This suggests **inductive bias vs conditioning tradeoff**.

3. **ROI vs full-volume improvement:** Larger improvements in ROI (brain tissue) than full volume suggest FiLM is most beneficial in high-SNR, anatomically complex regions where gradient direction matters most.

4. **Restormer FA anomaly:** Degraded FA with FiLM in Restormer may indicate:
   - Overfitting to bvec metadata (test-set bvecs similar to train?)
   - Suboptimal FiLM placement for attention layers (should condition attention weights instead of feature maps?)
   - Need for architecture-specific hyperparameter tuning (e.g., different film_hidden_dim)

5. **Computational cost:** 3-4% inference overhead is clinically acceptable. Training time increase (2.6-3.6×) is a one-time cost and may be reduced with better initialization or learning rate schedules.

6. **Alternative conditioning strategies:** Future work could explore:
   - Attention-based conditioning for transformers
   - Learnable positional embeddings (like ViT) instead of explicit bvecs
   - Cross-attention between gradient metadata and features

---

## Writing Style Recommendations

1. **Be concise:** This is an ablation, not the main contribution. Aim for ~1-1.5 pages total including figure/table.

2. **Be honest about limitations:** Don't hide the Restormer FA result - frame it as "architecture-specific behavior requiring further investigation" rather than a failure.

3. **Focus on ROI metrics:** These are more clinically relevant than full-volume metrics (which include background).

4. **Avoid overselling:** FiLM is a **modest but consistent** improvement, not a breakthrough. Frame as "simple, low-overhead mechanism that complements RGS diversity" rather than "essential component".

5. **Emphasize generality:** FiLM is a general conditioning framework, not specific to DWMRI (cite Perez et al. 2018 and note its use in VQA, image generation, etc.).

---

## Related Work / Citations

**FiLM original paper:**
- Perez, E., Strub, F., De Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. AAAI 2018.

**Similar conditioning in medical imaging:**
- (Search for) Conditional normalization in MRI reconstruction (e.g., fastMRI with acquisition parameter conditioning)
- (Search for) Positional encodings in 3D medical image segmentation/reconstruction

**Related DWMRI work:**
- Note if any prior DWMRI denoising work uses gradient direction as input (unlikely, but worth checking)
- Position this as first application of FiLM to DWMRI denoising (as far as you know)

---

## Implementation/Reproducibility Notes (for Methods section or appendix)

- **Code availability:** `src/utils/film_layer.py` (FiLMLayer implementation), model wiring in `src/*/model.py`
- **Configuration flag:** `model.use_film_conditioning=true`, `model.film_hidden_dim=32`
- **Training:** No special hyperparameter tuning beyond baseline RGS (same LR, batch size, epochs)
- **Inference:** Same `n_context_samples` and `n_preds` as baseline (FiLM conditioning is applied automatically when `orientation_info` is provided)

---

## Summary Checklist for Paper Writeup

- [ ] 1-paragraph motivation clearly stating the question
- [ ] 1-paragraph description of discarded naive approach + why
- [ ] 2-3 paragraphs on FiLM design, placement, and implementation
- [ ] 1-2 paragraphs on D-Brain quantitative results with Table 1
- [ ] 1 paragraph on Stanford generalization
- [ ] Figure 1: FA/MD comparison (Noisy, Baseline, FiLM, GT) on D-Brain
- [ ] (Optional) Figure 2: Stanford FA/MD qualitative results
- [ ] Discussion points on architecture dependency, ROI improvements, Restormer anomaly
- [ ] Related work citations (Perez et al. 2018 at minimum)
- [ ] Reproducibility notes (config flags, code locations)

**Target length:** 1-1.5 pages text + 1 table + 1-2 figures = ~2-3 pages total

**Positioning:** Frame as "natural extension to test whether explicit conditioning helps" rather than "critical missing component". Let the results (1.5 dB PSNR improvement in DRCNet) speak for themselves.
