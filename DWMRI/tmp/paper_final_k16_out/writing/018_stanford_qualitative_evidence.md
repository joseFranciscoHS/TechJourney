# Point 6: Stanford HARDI Qualitative Evidence

**Context**: Point 6 from `pruebas_faltantes_todos_20260628.md` — Stanford HARDI has no ground truth, so quantitative claims are limited. Need visual qualitative evidence.

**Problem**: The paper text (line 411, line 461) has TODOs requesting visual panels or ROI variance summaries for Stanford.

---

## Current Status in Paper

### TODO line 411 (§4.5 Stanford Generalization):
> "Add visual Stanford FA/MD panels, residual maps, and/or homogeneous-ROI variance summaries. Without those panels, keep Stanford claims limited to feasibility and scalability."

### TODO line 491 (Stanford FA-map figure):
> "Replace the planned Stanford HARDI FA-map comparison with final image panels or remove the figure before submission. The final figure should show a representative axial slice crossing major white-matter tracts for noisy input, Patch2Self, MD-S2S, DRCNet-Hybrid-RGS, and Restormer-Hybrid-RGS."

**Current paper claim** (line 453-461):
- Emphasizes qualitative assessment, FA/MD plausibility, scalability
- Mentions FiLM FA-MAE values (0.0356 DRCNet, 0.0756 Restormer) vs "internal reference" (not GT)
- No figures currently included

---

## Options for Stanford Evidence

### Option A: Add 1 FA Map Comparison Panel (Minimum Viable)

**What to generate**:
- Single axial slice through corpus callosum / corona radiata
- Columns: Noisy input | DRCNet-RGS | Restormer-RGS | (optional: Patch2Self, MP-PCA)
- Color-coded FA map (0.0 = black, 1.0 = bright)
- Annotations: corpus callosum (high FA ~0.6-0.8), gray matter (low FA ~0.1-0.2)

**Narrative to add**:
> "Figure X shows FA maps for a representative axial slice from Stanford HARDI. The denoised FA maps preserve high anisotropy in major white-matter tracts (corpus callosum, corona radiata) while suppressing speckled noise in gray matter and CSF. Coherent fiber structures remain anatomically plausible, supporting the feasibility of Hybrid RGS on real scanner data."

**Effort**: ~30 min - 1 hr (if `.npy` maps exist)

**Pros**:
- Minimal, defensible claim
- No quantitative overreach
- Satisfies reviewer expectation for visual evidence

**Cons**:
- Still qualitative only
- No baseline comparison if Patch2Self/MP-PCA Stanford runs don't exist

---

### Option B: Add Residual Maps

**What to generate**:
- Residual = Noisy input - Denoised output
- If denoising is good, residual should look like random noise (no edges)
- If denoising removes anatomy, residual will show structured brain features

**Narrative to add**:
> "Figure X shows residual maps (noisy input minus denoised output) for Stanford HARDI. The residuals resemble spatially uncorrelated noise without visible anatomical boundaries, indicating that Hybrid RGS preserves brain structure while removing scanner noise."

**Effort**: ~30 min

**Pros**:
- Strong qualitative evidence that denoising doesn't remove anatomy
- Commonly used in image restoration papers
- Easy to compute (just subtract arrays)

**Cons**:
- Still no quantitative metric
- Requires careful colormap choice (perceptually uniform, e.g., `viridis`)

---

### Option C: Variance in Homogeneous ROIs

**What to measure**:
- Select homogeneous white-matter ROI (e.g., centrum semiovale)
- Compute `std(FA)` within that ROI for each method
- Lower variance = better denoising in uniform regions

**Narrative to add**:
> "Table X reports FA standard deviation within a homogeneous white-matter ROI (centrum semiovale). Hybrid RGS reduces intra-ROI variance by X% compared to noisy input, indicating suppression of random fluctuations while preserving mean tissue properties."

**Effort**: ~1 hr (requires FA map computation + ROI segmentation)

**Pros**:
- Pseudo-quantitative metric (not ground truth, but interpretable)
- Common in no-reference denoising evaluation
- Can compare methods even without GT

**Cons**:
- Requires manual ROI selection or atlas-based segmentation
- Variance reduction alone doesn't prove anatomical accuracy

---

### Option D: Remove Stanford Figures, Keep Text Claims Limited

**What to do**:
- Remove both TODOs (lines 411, 491)
- Keep text claims limited to "feasibility" and "qualitative plausibility"
- Reference FiLM FA-MAE values as stability metrics (not ground-truth errors)

**Narrative to keep**:
> "Qualitatively, the denoised Stanford volumes show clear suppression of granular scanner noise while preserving major anatomical boundaries... FA maps after denoising preserve coherent high-anisotropy structures such as the corpus callosum and corona radiata without artificially inflating anisotropy in gray matter or CSF."

**Effort**: 0 (just remove TODOs)

**Pros**:
- Honest about limitations
- No misleading visual claims
- Focuses on controlled D-Brain experiments

**Cons**:
- Reviewers may ask for Stanford evidence
- Weakens generalization claim
- Makes Stanford experiment seem incomplete

---

## Recommended Approach (For Deadline)

**Minimum viable (1-2 hours total)**:
1. ✅ **Option A: 1 FA map comparison panel** (DRCNet, Restormer, noisy input)
2. ✅ **Option B: Residual maps** for same slice
3. ❌ Skip Option C (variance ROI) — nice-to-have, not critical
4. ✅ Update §4.5 text to reference Figure X

**Panel layout** (2×3 grid):

| Row | Column 1 | Column 2 | Column 3 |
|-----|----------|----------|----------|
| **Top** | Noisy FA | DRCNet FA | Restormer FA |
| **Bottom** | — | DRCNet residual | Restormer residual |

**Figure caption**:
> "Qualitative evaluation on Stanford HARDI. Top row: FA maps from noisy input, DRCNet-Hybrid-RGS, and Restormer-Hybrid-RGS. High-anisotropy white-matter tracts (corpus callosum, corona radiata) are preserved. Bottom row: Residual maps (noisy - denoised) show spatially uncorrelated noise patterns without anatomical structure, indicating that denoising preserves brain features."

---

## Implementation Steps

### 1. Verify FA/MD Maps Exist

```bash
# Check if .npy maps are available
ls tmp/paper_final_k16_out/drcnet_hybrid_rgs/metrics/stanford_film_conditioning/

# Expected files:
# - fa_map.npy
# - md_map.npy
# - ad_map.npy
# - rd_map.npy
# - denoised_volumes.npy (or similar)
```

**If missing**: Run DTI tensor fitting on Stanford denoised outputs:
```bash
# Example command (adjust paths)
python src/paper_eval/dti_metrics.py \
  --denoised-path tmp/.../stanford_film_conditioning/denoised_dwi.npy \
  --output-dir tmp/.../stanford_film_conditioning/
```

---

### 2. Generate FA Map Visualization

**Python script** (save as `scripts/visualize_stanford_fa.py`):

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE = Path("tmp/paper_final_k16_out")
NOISY_FA = BASE / "stanford_noisy/fa_map.npy"  # if available
DRCNET_FA = BASE / "drcnet_hybrid_rgs/metrics/stanford_film_conditioning/fa_map.npy"
RESTORMER_FA = BASE / "restormer_hybrid_rgs/metrics/stanford_film_conditioning/fa_map.npy"

# Load
noisy_fa = np.load(NOISY_FA) if NOISY_FA.exists() else None
drcnet_fa = np.load(DRCNET_FA)
restormer_fa = np.load(RESTORMER_FA)

# Select slice (axial, through corpus callosum)
slice_idx = drcnet_fa.shape[2] // 2  # Middle slice, adjust as needed

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

if noisy_fa is not None:
    axes[0].imshow(noisy_fa[:, :, slice_idx], cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Noisy Input FA')
else:
    axes[0].text(0.5, 0.5, 'Noisy FA\nNot Available', 
                 ha='center', va='center', transform=axes[0].transAxes)

axes[1].imshow(drcnet_fa[:, :, slice_idx], cmap='hot', vmin=0, vmax=1)
axes[1].set_title('DRCNet-Hybrid-RGS FA')

axes[2].imshow(restormer_fa[:, :, slice_idx], cmap='hot', vmin=0, vmax=1)
axes[2].set_title('Restormer-Hybrid-RGS FA')

for ax in axes:
    ax.axis('off')

plt.colorbar(axes[2].images[0], ax=axes, fraction=0.02, pad=0.04, label='FA')
plt.tight_layout()
plt.savefig('tmp/paper_final_k16_out/figures/stanford_fa_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: tmp/paper_final_k16_out/figures/stanford_fa_comparison.png")
```

---

### 3. Generate Residual Maps

**Add to script**:

```python
# Load denoised DWI volumes (single example volume)
NOISY_DWI = BASE / "stanford_noisy/noisy_dwi.npy"  # shape: (H, W, D, G)
DRCNET_DWI = BASE / "drcnet_hybrid_rgs/metrics/stanford_film_conditioning/denoised_dwi.npy"
RESTORMER_DWI = BASE / "restormer_hybrid_rgs/metrics/stanford_film_conditioning/denoised_dwi.npy"

noisy_dwi = np.load(NOISY_DWI)
drcnet_dwi = np.load(DRCNET_DWI)
restormer_dwi = np.load(RESTORMER_DWI)

# Select one gradient volume (e.g., first b=2000 volume)
vol_idx = 10  # Adjust as needed
slice_idx = noisy_dwi.shape[2] // 2

# Compute residuals
drcnet_residual = noisy_dwi[:, :, slice_idx, vol_idx] - drcnet_dwi[:, :, slice_idx, vol_idx]
restormer_residual = noisy_dwi[:, :, slice_idx, vol_idx] - restormer_dwi[:, :, slice_idx, vol_idx]

# Plot residuals
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(drcnet_residual, cmap='viridis')
axes[0].set_title('DRCNet Residual')
axes[0].axis('off')

axes[1].imshow(restormer_residual, cmap='viridis')
axes[1].set_title('Restormer Residual')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('tmp/paper_final_k16_out/figures/stanford_residuals.png', dpi=300, bbox_inches='tight')
print("Saved: tmp/paper_final_k16_out/figures/stanford_residuals.png")
```

---

### 4. Update Paper Text

**Replace TODO line 411** (in §4.5):

```latex
Qualitatively, the denoised Stanford volumes show clear suppression of granular 
scanner noise while preserving major anatomical boundaries, including the ventricles 
and gray--white matter interfaces. Figure~\ref{fig:stanford_fa} shows FA maps for a 
representative axial slice. High-anisotropy white-matter tracts such as the corpus 
callosum and corona radiata are preserved without artificial inflation of FA in 
gray matter or CSF. Residual maps (noisy input minus denoised output) show spatially 
uncorrelated noise patterns without visible anatomical structure, indicating that 
denoising removes scanner noise while preserving brain features.
```

**Add figure reference** (after line 461):

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=\linewidth]{figures/stanford_fa_comparison.png}
  \caption{Qualitative Stanford HARDI FA maps. Top row: noisy input, DRCNet-Hybrid-RGS, 
  Restormer-Hybrid-RGS. Denoised FA maps preserve high anisotropy in corpus callosum and 
  corona radiata. Bottom row: residual maps (noisy - denoised) resemble spatially 
  uncorrelated noise, indicating preservation of anatomical features.}
  \label{fig:stanford_fa}
\end{figure}
```

**Remove TODO line 491** (Stanford FA-map comparison) — replaced by Figure~\ref{fig:stanford_fa}.

---

## Validation Checklist

After generating figures:

- [ ] FA maps show recognizable brain anatomy (corpus callosum, ventricles)
- [ ] High FA (~0.6-0.8) in white-matter tracts, low FA (~0.1-0.2) in gray matter
- [ ] Residuals do NOT contain visible edges or anatomical structures
- [ ] Colormap is perceptually uniform (`hot` for FA, `viridis` for residuals)
- [ ] Figure caption explains what to look for
- [ ] Paper text references Figure~\ref{fig:stanford_fa}
- [ ] Both TODOs (lines 411, 491) are removed

---

## If FA Maps Don't Exist

### Option 1: Generate them from denoised DWI

```bash
# Run DTI fitting on Stanford denoised outputs
python src/paper_eval/compute_stanford_fa.py \
  --drcnet-dwi tmp/.../drcnet_hybrid_rgs/.../denoised_dwi.npy \
  --restormer-dwi tmp/.../restormer_hybrid_rgs/.../denoised_dwi.npy \
  --output-dir tmp/.../figures/
```

### Option 2: Use DIPY directly

```python
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
import numpy as np

# Load denoised DWI
denoised_dwi = np.load("denoised_dwi.npy")  # shape: (H, W, D, G)

# Load bvals/bvecs (Stanford HARDI)
from dipy.data import get_fnames
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

# Fit tensor
tenmodel = TensorModel(gtab)
tenfit = tenmodel.fit(denoised_dwi)

# Extract FA
fa_map = tenfit.fa
np.save("fa_map.npy", fa_map)
```

---

## Timeline

| Task | Effort | Priority |
|------|--------|----------|
| Verify FA/MD maps exist | 5 min | High |
| Generate FA comparison figure | 30 min | High |
| Generate residual maps | 20 min | High |
| Update paper text (§4.5) | 15 min | High |
| Add figure + caption | 10 min | High |
| **Total** | **~1.5 hrs** | — |

---

## Decision Required

**Choose one**:

1. ✅ **Recommended**: Generate FA + residual figures (1.5 hrs) — strengthens generalization claim
2. ⚠️ **Alternative**: Remove Stanford figures, keep claims limited to "feasibility" (0 hrs, but weaker)

**For deadline pressure**: Option 1 is recommended if FA maps already exist or can be generated quickly.

---

## Contact

If FA maps are missing or denoised DWI files can't be located, ask:
- Where are Stanford FiLM denoised outputs stored?
- Do we have Patch2Self / MP-PCA Stanford baselines for comparison?
- Should we generate just DRCNet/Restormer or include baselines?
