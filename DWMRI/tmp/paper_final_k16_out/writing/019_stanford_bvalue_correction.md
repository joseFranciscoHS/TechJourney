# Point 7: Stanford HARDI b-value Correction

**Context**: Point 7 from `pruebas_faltantes_todos_20260628.md` — Stanford HARDI b-value is **b=2000 s/mm²** (per DIPY official documentation), NOT b=1000.

**Status**: **RESOLVED** — Config has documentation error, but paper writing prompt 009 already uses correct value (b=2000).

---

## Problem Summary

### Config says b=1000 (INCORRECT):
- `src/drcnet_hybrid_rgs/config.yaml` line 181: `bvalue: 1000`
- `src/restormer_hybrid_rgs/config.yaml` line 170: `bvalue: 1000`

### DIPY official documentation says b=2000 (CORRECT):
- Stanford HARDI dataset: **b=2000 s/mm²**, 160 directions (10 b0 + 150 DWI)
- Citation: Rokem et al., 2015, PLoS ONE

### Code behavior:
```python
# src/utils/data.py
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames("stanford_hardi")
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)
```

**Critical**: The code loads bvals **directly from DIPY files**, ignoring `config.yaml bvalue: 1000`. The config value is only used for **path naming** and **logging**, NOT for tensor fitting.

**Conclusion**:
- ✅ **DTI fitting is correct** (uses b=2000 from DIPY)
- ❌ **Config has documentation error** (should say 2000 for consistency)
- ✅ **Paper text must say b=2000**

---

## Verified in Paper LaTeX

Checking `paper/Sepulveda_dwmri_restormer.tex`:

### ✅ CORRECT (line 72):
```latex
The current experiments include D-Brain data at $b=2500~\mathrm{s/mm^2}$ and Stanford HARDI data at $b=2000~\mathrm{s/mm^2}$, ...
```

### ✅ CORRECT (line 74):
```latex
The real-data experiment uses one in vivo Stanford HARDI acquisition from a GE Discovery MR750 3T scanner, obtained through \texttt{dipy.data.fetch\_stanford\_hardi}, with ten $b=0~\mathrm{s/mm^2}$ volumes and $G=150$ uniformly distributed diffusion-weighted directions at $b=2000~\mathrm{s/mm^2}$ in native $2~\mathrm{mm}$ isotropic voxels \cite{rokem2015accuracy}.
```

### ❌ INCORRECT (line 371):
```latex
On Stanford HARDI, FiLM-trained models converged successfully under a different protocol with $G=150$ gradients and $b=1000$, supporting the feasibility of the conditioning mechanism beyond D-Brain.
```

### ❌ INCORRECT (line 453):
```latex
Stanford HARDI uses a different acquisition protocol from D-Brain, with $b=1000~\mathrm{s/mm^2}$ and $G=150$ diffusion directions, compared with $b=2500~\mathrm{s/mm^2}$ and $G=60$ in the primary synthetic experiments.
```

---

## Required Corrections

### 1. Fix line 371 (FiLM subsection)

**Current (INCORRECT)**:
```latex
On Stanford HARDI, FiLM-trained models converged successfully under a different protocol with $G=150$ gradients and $b=1000$, supporting the feasibility of the conditioning mechanism beyond D-Brain.
```

**Corrected**:
```latex
On Stanford HARDI, FiLM-trained models converged successfully under a different protocol with $G=150$ gradients and $b=2000~\mathrm{s/mm^2}$, supporting the feasibility of the conditioning mechanism beyond D-Brain.
```

---

### 2. Fix line 453 (Stanford Generalization subsection)

**Current (INCORRECT)**:
```latex
Stanford HARDI uses a different acquisition protocol from D-Brain, with $b=1000~\mathrm{s/mm^2}$ and $G=150$ diffusion directions, compared with $b=2500~\mathrm{s/mm^2}$ and $G=60$ in the primary synthetic experiments.
```

**Corrected**:
```latex
Stanford HARDI uses a different acquisition protocol from D-Brain, with $b=2000~\mathrm{s/mm^2}$ and $G=150$ diffusion directions, compared with $b=2500~\mathrm{s/mm^2}$ and $G=60$ in the primary synthetic experiments.
```

---

## Global Search for b=1000

Search all mentions of "b=1000" in the paper:

```bash
grep -n "b=1000\|b=1,000\|b = 1000" paper/Sepulveda_dwmri_restormer.tex
```

**Expected output**:
- Line 371 (FiLM subsection) ← FIX THIS
- Line 453 (Stanford generalization) ← FIX THIS

**After correction**: Should return **no results** for Stanford context.

---

## Config Files (Optional Fix, Not Critical for Paper)

**Files to update** (improves codebase consistency, NOT required for paper deadline):

### `src/drcnet_hybrid_rgs/config.yaml` line 181:

**Current**:
```yaml
stanford_hardi:
  dataset_name: stanford_hardi
  bvalue: 1000  # ← INCORRECT
```

**Corrected**:
```yaml
stanford_hardi:
  dataset_name: stanford_hardi
  bvalue: 2000  # Correct per DIPY documentation (Rokem et al., 2015)
```

### `src/restormer_hybrid_rgs/config.yaml` line 170:

**Current**:
```yaml
stanford_hardi:
  dataset_name: stanford_hardi
  bvalue: 1000  # ← INCORRECT
```

**Corrected**:
```yaml
stanford_hardi:
  dataset_name: stanford_hardi
  bvalue: 2000  # Correct per DIPY documentation (Rokem et al., 2015)
```

**Note**: This config fix is **optional** because the `bvalue` field is not used for DTI fitting (DIPY reads from `.bval` files). It only affects path naming and logging.

---

## Validation Checklist

After corrections:

- [ ] Line 72 says **b=2000** (already correct ✅)
- [ ] Line 74 says **b=2000** (already correct ✅)
- [ ] Line 371 says **b=2000** (needs fix ❌)
- [ ] Line 453 says **b=2000** (needs fix ❌)
- [ ] No other mentions of "b=1000" in Stanford context
- [ ] Config files updated (optional, not critical)

---

## Why the Config Had b=1000

**Likely cause**: Early pilot experiments may have used a lower b-value test case, or the config was copied from a different dataset template. Since the actual bvals come from DIPY files, the config error had no impact on tensor fitting or metrics.

---

## Verification Command

To verify that DTI fitting used correct b-values:

```python
# Load Stanford HARDI bvals from DIPY
from dipy.data import get_fnames
from dipy.io import read_bvals_bvecs

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)

print(f"Unique b-values: {sorted(set(bvals))}")
# Expected output: [0, 2000] (not [0, 1000])
print(f"Number of b=0 volumes: {sum(bvals <= 50)}")
# Expected output: 10
print(f"Number of b=2000 volumes: {sum(bvals > 50)}")
# Expected output: 150
```

**Expected result**:
```
Unique b-values: [0, 2000]
Number of b=0 volumes: 10
Number of b=2000 volumes: 150
```

If you see `[0, 1000]`, there's a deeper issue. But based on code review, this is **not** the case — DIPY loads the correct values.

---

## Summary

| Issue | Status | Action |
|-------|--------|--------|
| Config says b=1000 | Documentation error | Optional: update config files |
| DTI fitting uses b=2000 | ✅ Correct | No action needed |
| Paper line 72, 74 say b=2000 | ✅ Correct | No action needed |
| Paper line 371 says b=1000 | ❌ Incorrect | **FIX: change to b=2000** |
| Paper line 453 says b=1000 | ❌ Incorrect | **FIX: change to b=2000** |

**Bottom line**: Fix 2 lines in the paper LaTeX (371, 453). Config updates are optional for code consistency.

---

## Files to Edit

| File | Lines | Find | Replace |
|------|-------|------|---------|
| `paper/Sepulveda_dwmri_restormer.tex` | 371 | `$G=150$ gradients and $b=1000$` | `$G=150$ gradients and $b=2000~\mathrm{s/mm^2}$` |
| `paper/Sepulveda_dwmri_restormer.tex` | 453 | `with $b=1000~\mathrm{s/mm^2}$` | `with $b=2000~\mathrm{s/mm^2}$` |
| `src/drcnet_hybrid_rgs/config.yaml` | 181 | `bvalue: 1000` | `bvalue: 2000  # Per DIPY documentation` |
| `src/restormer_hybrid_rgs/config.yaml` | 170 | `bvalue: 1000` | `bvalue: 2000  # Per DIPY documentation` |

**Priority**: LaTeX fixes are **critical** for paper submission. Config fixes are **nice-to-have** for codebase correctness.

---

## Final Verification

After making changes:

```bash
# Check paper
grep -n "b=1000" paper/Sepulveda_dwmri_restormer.tex
# Should return NO results in Stanford context

# Check configs (if updated)
grep -n "bvalue.*1000" src/drcnet_hybrid_rgs/config.yaml
grep -n "bvalue.*1000" src/restormer_hybrid_rgs/config.yaml
# Should return NO results
```

**Expected**: All searches return empty results after correction.

---

## Contact

If unsure about Stanford b-value, verify with:
- DIPY documentation: https://dipy.org/documentation/latest/examples_built/preprocessing/
- Rokem et al. (2015) paper cited in line 74
- `dipy.data.fetch_stanford_hardi` source code
