# Paper Correction Prompts — Index

**Date**: July 1, 2026  
**Context**: TODOs and corrections from `experiments/pruebas_faltantes_todos_20260701.md` (successor to 20260628)

This directory contains detailed prompts for correcting critical issues in the paper before submission.

---

## Correction Prompts

| # | Point | Prompt File | Status | Priority | Effort |
|---|-------|-------------|--------|----------|--------|
| **1** | Diffusivity Units | [`015_diffusivity_units_correction.md`](015_diffusivity_units_correction.md) | Done (paper) | 🔴 Critical | — |
| **2** | FiLM Metrics | [`016_film_metrics_correction.md`](016_film_metrics_correction.md) | Done (paper + aux) | 🔴 Critical | — |
| **3** | Registry Conflict | [`017_registry_conflict_documentation.md`](017_registry_conflict_documentation.md) | Acknowledged | 🟢 Info | — |
| **6** | Stanford Evidence | [`018_stanford_qualitative_evidence.md`](018_stanford_qualitative_evidence.md) | Deferred | 🟡 High | 1-2 hrs |
| **7** | Stanford b-value | [`019_stanford_bvalue_correction.md`](019_stanford_bvalue_correction.md) | Done (paper + configs) | 🔴 Critical | — |
| **8** | K=16 Rerun | [`020_k16_rerun_results_update.md`](020_k16_rerun_results_update.md) | Ready | 🔴 Critical | 30-45 min |

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
**Solution**: Superseded by **Point 8** (June 2026 rerun). For historical context see [`017_registry_conflict_documentation.md`](017_registry_conflict_documentation.md).  
**Status**: Resolved via rerun — use `tmp/paper_final_k16_rerun_20260628T042410Z/registry.jsonl`

---

### Point 8: K=16 Canonical Rerun (CRITICAL)
**Problem**: Paper cites May-2026 canonical baselines; mixed values (e.g. 23.82 vs 23.93 in main vs FiLM table)  
**Solution**: Execute [`020_k16_rerun_results_update.md`](020_k16_rerun_results_update.md) against `paper/Sepulveda_dwmri_restormer.tex`  
**Source**: `tmp/paper_final_k16_rerun_20260628T042410Z/` (4 jobs: `*_rgs_final` D-Brain + Stanford)  
**Scope**: Main comparison, all K=16 canonical rows in ablation tables, FiLM baseline + Δ rows, registry provenance, implementation seeds  
**Key new values**:
| Metric | DRCNet | Restormer |
|--------|--------|-----------|
| PSNR-ROI (D-Brain) | **26.93** | **23.43** |
| FA-MAE (D-Brain) | 0.2575 | **0.2238** (now better than DRCNet) |
| FiLM Δ ROI | **−0.69 dB** | **−0.74 dB** (recomputed vs new baseline) |
| Stanford noisy PSNR-ROI K=16 | **38.07** | **26.05** |

**Not rerun**: FiLM conditioned absolute values, baselines, K≠16 ablations

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
4. ⬜ **Point 8** (K=16 rerun) — Sync paper with `paper_final_k16_rerun` registry → 30-45 min

### Phase 2: Documentation (5 min)
5. ✅ **Point 3** (Registry conflict) — Superseded by Point 8 rerun

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
`paper_metrics_summary.csv` in the **old** output tree may contain corrupted baselines. The June 2026 rerun (`tmp/paper_final_k16_rerun_20260628T042410Z/`) provides clean canonical values. Use prompt **020** to update the paper.

---

## Contact

For questions about:
- **Diffusivity units** → See registry `dti_sanity_gt.md_range` (max ≈ 879,827 phantom units)
- **FiLM values** → Baseline from rerun registry; FiLM conditioned from old registry lines 131-132
- **K=16 canonical baselines** → `tmp/paper_final_k16_rerun_20260628T042410Z/registry.jsonl`
- **Registry conflicts** → Resolved by rerun; see Point 8 / prompt 020
- **Stanford b-value** → See DIPY documentation (Rokem et al., 2015)

---

## Progress Tracking

- [x] Point 1 prompt created
- [x] Point 2 prompt created
- [x] Point 3 documentation created
- [x] Point 6 prompt created
- [x] Point 7 prompt created
- [x] Point 8 prompt created
- [x] Point 1 executed (paper — footnotes, TODOs removed)
- [x] Point 2 executed (paper + orientation_conditioning_metrics_summary.md + 001_orientation_conditioning.md)
- [x] Point 3 read/acknowledged
- [ ] Point 6 decision made / executed (Stanford figures — deferred to `pruebas_faltantes_todos_20260701.md` items #5–#6)
- [x] Point 7 executed (paper + config.yaml)
- [ ] Point 8 executed (paper sync with K=16 rerun)

---

**Last updated**: 2026-07-01 (Point 8 prompt added)  
**Author**: AI assistant (based on `pruebas_faltantes_todos_20260701.md`)
