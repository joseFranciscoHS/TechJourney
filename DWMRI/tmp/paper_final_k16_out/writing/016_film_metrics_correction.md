# Point 2: FiLM Metrics Correction

**Context**: The `orientation_conditioning_metrics_summary.md` file contains incorrect/inflated PSNR-ROI values that don't match the authoritative `registry.jsonl`.

**Problem**: Line 59-60 of `pruebas_faltantes_todos_20260628.md` identifies the issue.

---

## Incorrect Values (in orientation_conditioning_metrics_summary.md)

### D-Brain, σ=0.1, K=16

**DRCNet** (lines 17-30):
- ❌ PSNR (ROI): **30.40 dB** (INCORRECT - inflated)
- ❌ Baseline PSNR (ROI): **28.48 dB** (appears copied from MP-PCA, not from actual baseline)

**Restormer** (lines 42-55):
- ❌ PSNR (ROI): **27.94 dB** (INCORRECT - inflated)
- ❌ Baseline PSNR (ROI): **26.56 dB** (may be incorrect)

**Stanford** (lines 72-94):
- ❌ DRCNet PSNR (ROI): **45.06 dB** (INCORRECT - extremely inflated)
- ❌ Restormer PSNR (ROI): **32.56 dB** (INCORRECT)

---

## Correct Values (from registry.jsonl)

### D-Brain Baselines (σ=0.1, K=16)

**Registry lines 12-13** (authoritative baseline values):

| Architecture | PSNR (full) | SSIM (full) | PSNR-ROI | FA-MAE | MD-MAE | Params |
|--------------|-------------|-------------|----------|---------|---------|---------|
| **DRCNet baseline** | 23.927 | 0.4404 | **26.882** | 0.2587 | 3449.38 | 116,002 |
| **Restormer baseline** | 22.835 | 0.4271 | **23.220** | 0.2378 | 3454.58 | 177,883 |

### D-Brain FiLM (σ=0.1, K=16)

**Registry lines 131-134** (authoritative FiLM values):

| Architecture | PSNR (full) | SSIM (full) | PSNR-ROI | FA-MAE | MD-MAE | Params |
|--------------|-------------|-------------|----------|---------|---------|---------|
| **DRCNet FiLM** | 25.403 | 0.4552 | **26.244** | 0.2405 | 3452.19 | 120,546 |
| **Restormer FiLM** | 23.603 | 0.4380 | **22.694** | 0.2603 | 3423.42 | 184,699 |

**Note**: Restormer FiLM has **worse PSNR-ROI** than baseline (22.694 vs 23.220) — a **-0.526 dB degradation**, not an improvement!

---

## Required Corrections

### 1. Update `orientation_conditioning_metrics_summary.md`

**File**: `tmp/paper_final_k16_out/orientation_conditioning_metrics_summary.md`

**DRCNet table (lines 17-30):**

Replace line 23:
```markdown
|| **PSNR (ROI)** | 28.48 dB | **30.40 dB** | **+1.92 dB** |
```

With:
```markdown
|| **PSNR (ROI)** | 26.88 dB | 26.24 dB | **-0.64 dB** |
```

**Restormer table (lines 42-55):**

Replace line 48:
```markdown
|| **PSNR (ROI)** | 26.56 dB | **27.94 dB** | **+1.38 dB** |
```

With:
```markdown
|| **PSNR (ROI)** | 23.22 dB | 22.69 dB | **-0.53 dB** |
```

**Summary table (lines 108-113):**

Replace:
```markdown
|| DRCNet | None | 23.93 | 28.48 | 0.2606 | 3539 | 116K | 34.0s |
|| DRCNet | **FiLM** | **25.40** | **30.40** | **0.2405** | **3452** | 120K | 35.3s |
|| Restormer | None | 22.83 | 26.56 | **0.2424** | 3502 | 178K | 126.2s |
|| Restormer | **FiLM** | **23.60** | **27.94** | 0.2603 | **3423** | 185K | 130.3s |
```

With:
```markdown
|| DRCNet | None | 23.93 | **26.88** | 0.2587 | 3449 | 116K | 34.0s |
|| DRCNet | **FiLM** | **25.40** | 26.24 | **0.2405** | **3452** | 120K | 35.3s |
|| Restormer | None | 22.84 | 23.22 | 0.2378 | 3455 | 178K | 126.2s |
|| Restormer | **FiLM** | **23.60** | 22.69 | **0.2603** | **3423** | 185K | 130.3s |
```

**Key takeaways section (lines 115-122):**

Update line 117:
```markdown
2. **DRCNet benefits more from FiLM than Restormer** (1.47 dB vs 0.77 dB improvement)
```

To:
```markdown
2. **FiLM improves full-volume PSNR** (+1.47 dB DRCNet, +0.77 dB Restormer) **but degrades ROI PSNR** (-0.64 dB DRCNet, -0.53 dB Restormer)
```

Add new bullet:
```markdown
3. **ROI degradation is unexpected** and requires investigation - possible causes: overfitting to background, FiLM placement, or identity initialization issues
```

### 2. Update prompt `001_orientation_conditioning.md` (if it references summary)

**File**: `tmp/paper_final_k16_out/writing/001_orientation_conditioning.md`

**Lines 51-52** mention D-Brain results. Update the prompt text to say:

```markdown
4. **Experimental Results** (2 paragraphs, ~200-250 words)
   - **D-Brain** quantitative results in a table (see metrics summary). 
   - **Full-volume improvement**: +1.47 dB PSNR and -28.8% MSE for DRCNet; +0.77 dB PSNR and -16.2% MSE for Restormer. 
   - **ROI metrics show degradation**: PSNR-ROI decreases by -0.64 dB (DRCNet) and -0.53 dB (Restormer).
   - **Tensor metrics**: FA-MAE improves in DRCNet (-7.7%) but worsens in Restormer (+7.4%); MD-MAE improves in both.
   - **Interpretation**: FiLM may improve global smoothness while degrading fine anatomical detail in ROI, or identity initialization may need tuning.
   - **Stanford** generalization: FiLM trains successfully on a different acquisition protocol (G=150, b=2000). No baseline comparison available; qualitative FA/MD assessment only.
   - Architecture dependency: Results suggest FiLM requires architecture-specific tuning.
```

### 3. Correct paper LaTeX text (if already written)

**File**: `paper/Sepulveda_dwmri_restormer.tex`

**Line 369** says:
```latex
For DRCNet, conditioning increases full-volume PSNR from $23.93$ to $25.40$ dB, a gain of $1.47$ dB, and increases ROI PSNR by $1.92$ dB. The ROI MSE decreases by $38.2\%$, ...
```

**Replace with**:
```latex
For DRCNet, conditioning increases full-volume PSNR from $23.93$ to $25.40$ dB, a gain of $1.47$ dB, but slightly decreases ROI PSNR from $26.88$ to $26.24$ dB ($-0.64$ dB). The full-volume MSE decreases by $28.8\%$, ...
```

**Line 369 also mentions Restormer** - update similarly:
```latex
For Restormer, FiLM improves full-volume PSNR by $0.77$ dB but decreases ROI PSNR from $23.22$ to $22.69$ dB ($-0.53$ dB). MD, AD, and RD errors improve. However, FA-MAE worsens by $7.4\%$, ...
```

---

## Critical Interpretation Change

### OLD narrative (incorrect):
> "FiLM improves PSNR-ROI by +1.92 dB (DRCNet) and +1.38 dB (Restormer), demonstrating consistent benefit."

### NEW narrative (correct):
> "FiLM improves **full-volume** PSNR consistently (+1.47 dB DRCNet, +0.77 dB Restormer), indicating better global denoising. However, **ROI PSNR degrades slightly** (-0.64 dB DRCNet, -0.53 dB Restormer), suggesting that the conditioning may smooth background more than brain tissue or that identity initialization requires tuning. Tensor metrics show mixed results: FA-MAE improves in DRCNet but worsens in Restormer, while MD-MAE improves in both. These results indicate that **FiLM is not universally beneficial** and requires careful architecture-specific design."

---

## Stanford FiLM Values (no correction needed)

The Stanford FiLM values in the summary are measured against noisy input (not GT), so the high PSNR values (45.06, 32.56) reflect **input preservation** rather than restoration quality. These should NOT be changed, but the interpretation should be clarified:

**Add note in summary** (after line 100):
```markdown
**Important**: Stanford PSNR values are measured against the noisy input (no GT available), so high PSNR indicates similarity to input, NOT denoising quality. Do not compare Stanford PSNR with D-Brain PSNR.
```

---

## Validation Checklist

After corrections:

- [ ] `orientation_conditioning_metrics_summary.md` shows correct registry values
- [ ] DRCNet baseline PSNR-ROI = **26.88** (not 28.48)
- [ ] DRCNet FiLM PSNR-ROI = **26.24** (not 30.40)
- [ ] Restormer baseline PSNR-ROI = **23.22** (not 26.56)
- [ ] Restormer FiLM PSNR-ROI = **22.69** (not 27.94)
- [ ] "Key takeaways" section reflects ROI degradation
- [ ] Prompt `001_orientation_conditioning.md` reflects corrected narrative
- [ ] Paper LaTeX text (if written) reflects degradation, not improvement
- [ ] Stanford values clarified as noisy-vs-noisy, not denoising quality

---

## Root Cause Analysis

**Why were values inflated?**

The incorrect values in `orientation_conditioning_metrics_summary.md` likely came from:
1. **Different metric paths** being merged incorrectly
2. **Baseline values copied from MP-PCA** (28.44 PSNR-ROI) instead of actual DRCNet baseline (26.88)
3. **Confusion between full-volume and ROI metrics**
4. **Registry line conflicts** (see Point #3) where inference_time_grid jobs overwrote baseline metrics

**Why is FiLM degrading ROI?**

Possible explanations:
1. **Identity initialization** may not be stable enough (γ≈1, β≈0 can drift)
2. **FiLM layers placed too early/late** in the backbone
3. **Overfitting to background** (FiLM sees more background voxels during training)
4. **Feature-space conditioning** may blur fine anatomical detail while improving global smoothness
5. **Training schedule** may need adjustment (longer warmup, lower LR for FiLM params)

**Action**: Investigate FiLM placement and initialization before claiming it as a recommended extension.

---

## Files to Update

| File | Lines | Action |
|------|-------|--------|
| `tmp/.../orientation_conditioning_metrics_summary.md` | 23, 48, 108-113, 117 | Replace inflated values with registry values |
| `tmp/.../writing/001_orientation_conditioning.md` | 51-52 | Update experimental results narrative |
| `paper/Sepulveda_dwmri_restormer.tex` | 369 | Correct FiLM results text (if already written) |

---

## Next Steps

1. **Update** `orientation_conditioning_metrics_summary.md` with correct values
2. **Revise** narrative in prompt `001_orientation_conditioning.md`
3. **Verify** `registry.jsonl` lines 12-13, 131-134 match the values above
4. **Decide** whether to include FiLM in paper given ROI degradation, or mark as "exploratory" with mixed results
5. **Investigate** why FiLM degrades ROI metrics before claiming it as a contribution
