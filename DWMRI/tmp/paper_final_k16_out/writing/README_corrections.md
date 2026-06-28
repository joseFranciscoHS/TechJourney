# Paper Correction Prompts — Index

**Date**: June 28, 2026  
**Context**: TODOs and corrections from `experiments/pruebas_faltantes_todos_20260628.md`

This directory contains detailed prompts for correcting critical issues in the paper before submission.

---

## Correction Prompts

| # | Point | Prompt File | Status | Priority | Effort |
|---|-------|-------------|--------|----------|--------|
| **1** | Diffusivity Units | [`015_diffusivity_units_correction.md`](015_diffusivity_units_correction.md) | Ready | 🔴 Critical | 10-15 min |
| **2** | FiLM Metrics | [`016_film_metrics_correction.md`](016_film_metrics_correction.md) | Ready | 🔴 Critical | 15-30 min |
| **3** | Registry Conflict | [`017_registry_conflict_documentation.md`](017_registry_conflict_documentation.md) | Documentation | 🟢 Info | 5 min read |
| **6** | Stanford Evidence | [`018_stanford_qualitative_evidence.md`](018_stanford_qualitative_evidence.md) | Ready | 🟡 Recommended | 1-2 hrs |
| **7** | Stanford b-value | [`019_stanford_bvalue_correction.md`](019_stanford_bvalue_correction.md) | Ready | 🔴 Critical | 5 min |

---

## Quick Summary

### Point 1: Diffusivity Units (CRITICAL)
**Problem**: MD-MAE, AD-MAE, RD-MAE are in arbitrary D-Brain phantom units, NOT mm²/s  
**Solution**: Add table footnotes explaining units, remove TODO comments  
**Files affected**: `paper/Sepulveda_dwmri_restormer.tex` (lines 206, 447, all tables with diffusivity errors)  
**Approach**: Option A — report as-is with disclaimers

---

### Point 2: FiLM Metrics (CRITICAL)
**Problem**: `orientation_conditioning_metrics_summary.md` has incorrect/inflated PSNR-ROI values  
**Solution**: Replace with authoritative values from `registry.jsonl` lines 12-13, 131-134  
**Critical finding**: FiLM **degrades** ROI PSNR (-0.64 dB DRCNet, -0.53 dB Restormer), NOT improves!  
**Files affected**: 
- `tmp/.../orientation_conditioning_metrics_summary.md`
- `tmp/.../writing/001_orientation_conditioning.md`
- `paper/Sepulveda_dwmri_restormer.tex` (line 369 if already written)

**Key corrections**:
| Metric | Old (incorrect) | New (correct) | Change |
|--------|-----------------|---------------|--------|
| DRCNet baseline ROI PSNR | 28.48 | **26.88** | - |
| DRCNet FiLM ROI PSNR | 30.40 | **26.24** | - |
| Restormer baseline ROI PSNR | 26.56 | **23.22** | - |
| Restormer FiLM ROI PSNR | 27.94 | **22.69** | - |

---

### Point 3: Registry Conflict (DOCUMENTATION ONLY)
**Problem**: Multiple registry entries for σ=0.1, K=16 due to `inference_time_grid` job overwriting metrics  
**Solution**: Use `registry.jsonl` lines 12-13 as canonical, ignore `paper_metrics_summary.csv` baseline rows  
**Status**: Already resolved, no action needed — just awareness for writing  
**Authoritative values**:
- DRCNet baseline: PSNR-ROI = **26.882**
- Restormer baseline: PSNR-ROI = **23.220**

---

### Point 6: Stanford Qualitative Evidence (RECOMMENDED)
**Problem**: Stanford HARDI has no ground truth, needs visual evidence  
**Solution**: Generate FA map + residual comparison figure (minimum viable)  
**Recommended approach**: 
1. 1 FA map panel (noisy, DRCNet, Restormer)
2. Residual maps (noisy - denoised)
3. Update §4.5 text, remove TODOs (lines 411, 491)

**Effort**: ~1.5 hrs if FA maps already exist

---

### Point 7: Stanford b-value (CRITICAL)
**Problem**: Paper mentions `b=1000` for Stanford in 2 places, correct value is `b=2000`  
**Solution**: Fix lines 371 and 453 in `paper/Sepulveda_dwmri_restormer.tex`  
**Status**: Lines 72 and 74 are already correct (say b=2000)  
**Config fix**: Optional — update `config.yaml` files (not critical for paper)

**Lines to fix**:
- Line 371: Change `$b=1000$` → `$b=2000~\mathrm{s/mm^2}$`
- Line 453: Change `$b=1000~\mathrm{s/mm^2}$` → `$b=2000~\mathrm{s/mm^2}$`

---

## Execution Order (Recommended)

### Phase 1: Critical Fixes (30-45 min)
1. ✅ **Point 7** (Stanford b-value) — 2 LaTeX line fixes → 5 min
2. ✅ **Point 1** (Diffusivity units) — Add table footnotes, remove TODOs → 15 min
3. ✅ **Point 2** (FiLM metrics) — Update summary + prompt, revise narrative → 20 min

### Phase 2: Documentation (5 min)
4. ✅ **Point 3** (Registry conflict) — Read for awareness, no edits needed → 5 min

### Phase 3: Figures (Optional, 1-2 hrs)
5. ⚠️ **Point 6** (Stanford evidence) — Generate FA/residual figures if time permits → 1.5 hrs

---

## Validation Commands

### After Phase 1:

```bash
# Check diffusivity TODOs removed
grep -n "TODO.*diffusivity\|TODO.*MD-MAE" paper/Sepulveda_dwmri_restormer.tex
# Should return NO results

# Check Stanford b-value corrected
grep -n "b=1000" paper/Sepulveda_dwmri_restormer.tex
# Should return NO results in Stanford context

# Check FiLM metrics updated
grep "30.40\|27.94\|28.48.*DRCNet.*baseline" \
  tmp/paper_final_k16_out/orientation_conditioning_metrics_summary.md
# Should return NO results (all replaced with correct values)
```

---

## Files to Edit (Summary)

### Critical (must fix before submission):
1. `paper/Sepulveda_dwmri_restormer.tex`
   - Lines 206, 447: Remove diffusivity TODO comments
   - Add footnotes after tables with MD-MAE/AD-MAE/RD-MAE
   - Lines 371, 453: Change b=1000 → b=2000
   - Line 369: Update FiLM results (if already written)

2. `tmp/.../orientation_conditioning_metrics_summary.md`
   - Lines 23, 48, 108-113, 117: Replace with registry values

3. `tmp/.../writing/001_orientation_conditioning.md`
   - Lines 51-52: Update experimental results narrative

### Optional (improves consistency):
4. `src/drcnet_hybrid_rgs/config.yaml` line 181: Change bvalue 1000 → 2000
5. `src/restormer_hybrid_rgs/config.yaml` line 170: Change bvalue 1000 → 2000

---

## Critical Findings Summary

### 🚨 FiLM ROI Degradation
The most significant discovery: **FiLM conditioning degrades ROI PSNR** while improving full-volume PSNR. This changes the narrative from "consistent improvement" to "mixed results requiring architecture-specific tuning."

**OLD claim** (incorrect):
> "FiLM improves PSNR-ROI by +1.92 dB (DRCNet) and +1.38 dB (Restormer)."

**NEW claim** (correct):
> "FiLM improves full-volume PSNR but slightly degrades ROI PSNR (-0.64 dB DRCNet, -0.53 dB Restormer), suggesting that conditioning may smooth background more than brain tissue or requires architecture-specific tuning."

### 📊 Diffusivity Units
D-Brain MD-MAE values (~3400-8500) are in **arbitrary phantom intensity units**, NOT mm²/s. Stanford HARDI values ARE in mm²/s. Must add disclaimers to avoid misinterpretation.

### 🔍 Registry Conflicts
`paper_metrics_summary.csv` contains corrupted baseline values due to `inference_time_grid` job overwriting metrics. Always use `registry.jsonl` directly.

---

## Contact

For questions about:
- **Diffusivity units** → See registry `dti_sanity_gt.md_range` (max ≈ 879,827 phantom units)
- **FiLM values** → See registry lines 12-13 (baseline), 131-134 (FiLM)
- **Registry conflicts** → See registry line 121 (corrupted inference_time_grid)
- **Stanford b-value** → See DIPY documentation (Rokem et al., 2015)

---

## Progress Tracking

- [x] Point 1 prompt created
- [x] Point 2 prompt created
- [x] Point 3 documentation created
- [x] Point 6 prompt created
- [x] Point 7 prompt created
- [ ] Point 1 executed
- [ ] Point 2 executed
- [ ] Point 3 read/acknowledged
- [ ] Point 6 decision made (generate or skip)
- [ ] Point 7 executed

---

**Last updated**: 2026-06-28  
**Author**: AI assistant (based on `pruebas_faltantes_todos_20260628.md`)
