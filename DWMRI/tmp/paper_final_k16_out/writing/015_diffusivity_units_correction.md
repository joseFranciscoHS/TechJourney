# Diffusivity Units Correction (Option A: Report with Disclaimer)

**Context**: Point 1 from `experiments/pruebas_faltantes_todos_20260628.md` — D-Brain diffusivity errors (MD-MAE, AD-MAE, RD-MAE) are reported in arbitrary phantom intensity units, NOT physical mm²/s.

**Approach**: Option A (recommended for deadline) — Keep current values but add clear disclaimers and footnotes.

---

## Problem Statement

Tables reporting MD-MAE, AD-MAE, and RD-MAE for D-Brain show values ~3400-8500, which are NOT physical mm²/s units:

- **D-Brain**: Digital phantom with arbitrary intensity scale (MD max ≈ 879,827)
- **Code behavior**: `src/paper_eval/dti_metrics.py` applies `invert_normalization` before tensor fitting, returning to phantom scale
- **Stanford HARDI**: Values ARE in physical mm²/s (DIPY computes from in-vivo data)

**Current TODO markers**:
- Line 206: "Daggered MD-MAE columns: confirm diffusivity-error units/scaling for MD-MAE, AD-MAE, and RD-MAE before submission."
- Line 447: "Daggered MD-MAE columns: confirm and report the diffusivity-error unit/scaling for MD-MAE, AD-MAE, and RD-MAE, for example $10^{-6}~\mathrm{mm^2/s}$ if that is the intended convention."

---

## Required Changes to `paper/Sepulveda_dwmri_restormer.tex`

### 1. Update Method Section (around line 144)

**Current text** (line 144):

```latex
with analogous MD-MAE, AD-MAE, and RD-MAE definitions. FA is dimensionless. MD, AD, and RD errors use the units returned by DIPY after denormalization; for D-Brain, these are relative diffusivity magnitudes from the phantom scale and should be compared across methods, not interpreted as physical diffusivities in $\mathrm{mm^2/s}$.
```

**Keep this as is** — it already explains the situation correctly.

---

### 2. Table~\ref{tab:objective_controlled_ablation} (lines 189-206)

**Current table header** (line 196):
```latex
Training objective & Angular & Masking & Random & PSNR-ROI $\uparrow$ & SSIM-ROI $\uparrow$ & FA-MAE $\downarrow$ & MD-MAE$^\dagger$ $\downarrow$ & Time/vol. (s) \\
```

**Current TODO** (line 206):
```latex
\TODO{Daggered MD-MAE columns: confirm diffusivity-error units/scaling for MD-MAE, AD-MAE, and RD-MAE before submission.}
```

**Action**:
- **Remove** the `\TODO{}` on line 206
- **Add** a table footnote immediately after `\end{table}` (after line 205) explaining the units:

```latex
\end{table}
\par\noindent\textit{Note:} $^\dagger$MD-MAE values are reported in arbitrary units derived from the D-Brain phantom intensity scale and should be interpreted as relative comparisons across methods.
```

---

### 3. Table~\ref{tab:main_comparison} (lines 426-447)

**Current table header** (line 433):
```latex
Method & PSNR $\uparrow$ & SSIM $\uparrow$ & MSE $\downarrow$ & PSNR-ROI $\uparrow$ & MSE-ROI $\downarrow$ & FA-MAE $\downarrow$ & MD-MAE$^\dagger$ $\downarrow$ & Time/vol. (s) \\
```

**Current TODO** (line 447):
```latex
\TODO{Daggered MD-MAE columns: confirm and report the diffusivity-error unit/scaling for MD-MAE, AD-MAE, and RD-MAE, for example $10^{-6}~\mathrm{mm^2/s}$ if that is the intended convention.}
```

**Action**:
- **Remove** the `\TODO{}` on line 447
- **Add** a table footnote immediately after `\end{table}` (after line 445) explaining the units:

```latex
\end{table}
\par\noindent\textit{Note:} $^\dagger$MD-MAE, AD-MAE, and RD-MAE for D-Brain experiments are reported in arbitrary units derived from the phantom intensity scale and should be interpreted as relative comparisons across methods. Stanford HARDI diffusivity errors are in physical mm$^2$/s units.
```

---

### 4. Other Tables with Diffusivity Errors

Review and add similar footnotes to any other tables that report MD-MAE, AD-MAE, or RD-MAE:

**Tables to check**:
- `\ref{tab:3d_vs_2d}` (line ~277-290) — has MD-MAE column
- `\ref{tab:film_ablation}` (line ~348-367) — has MD-MAE column  
- `\ref{tab:sampling_ablations}` (line ~224-251) — check if it reports diffusivity errors

**For each table with MD-MAE/AD-MAE/RD-MAE**:
- Keep the dagger symbol `$^\dagger$` in the column header if present
- Add the footnote after `\end{table}`:

```latex
\par\noindent\textit{Note:} $^\dagger$Diffusivity errors are in arbitrary D-Brain phantom units; compare across methods, not as physical mm$^2$/s values.
```

---

### 5. Do NOT Add Unit Labels to Column Headers

**Avoid** adding labels like these to D-Brain tables:
- ❌ `MD-MAE (×10⁻⁶ mm²/s)`
- ❌ `MD-MAE (μm²/s)`
- ❌ `MD-MAE (au)` (arbitrary units)

**Reason**: These would incorrectly suggest the values are in physical units or properly scaled. The dagger symbol `$^\dagger$` with a footnote is sufficient.

---

### 6. Stanford HARDI Tables

For Stanford HARDI results (if any tables report diffusivity errors):
- Add a footnote clarifying that **Stanford** values ARE in physical mm²/s:

```latex
\par\noindent\textit{Note:} Diffusivity errors for Stanford HARDI are in physical mm$^2$/s units.
```

---

## Summary of Edits

| Location | Line(s) | Action |
|----------|---------|--------|
| Method section | ~144 | **Keep as is** — already explains phantom units correctly |
| `\ref{tab:objective_controlled_ablation}` | 206 | **Delete TODO**, add footnote after line 205 |
| `\ref{tab:main_comparison}` | 447 | **Delete TODO**, add footnote after line 445 |
| `\ref{tab:3d_vs_2d}` | ~290 | Add footnote if MD-MAE present |
| `\ref{tab:film_ablation}` | ~367 | Add footnote if MD-MAE present |
| Other tables | Various | Add footnotes where MD-MAE/AD-MAE/RD-MAE appear |

---

## Validation Checklist

After making changes, verify:

- [ ] All `\TODO{}` comments about diffusivity units are removed
- [ ] Tables with MD-MAE/AD-MAE/RD-MAE have dagger symbols (`$^\dagger$`) in headers
- [ ] Each daggered table has a footnote explaining phantom vs. physical units
- [ ] NO unit labels (like "×10⁻⁶ mm²/s") are added to D-Brain table headers
- [ ] Method section (line ~144) still correctly explains the situation
- [ ] LaTeX compiles without errors
- [ ] Table footnotes render clearly below each table

---

## Rationale for Option A

**Why not Option B (re-scale to physical units)?**
- Requires determining calibration factor phantom → physical scale
- Needs re-computing DTI metrics for all runs (hours of compute)
- Risk of inconsistencies across run families
- **Deadline pressure**: Option A is safer and scientifically valid

**Scientific justification**:
- Cross-method comparison remains valid (same pipeline, same phantom)
- Stanford HARDI provides real-world mm²/s validation
- Many phantom studies report relative metrics without physical calibration
- The disclaimer makes interpretation clear to readers

---

## Example Footnote Templates

**Minimal (for single-metric tables)**:
```latex
\par\noindent\textit{Note:} $^\dagger$Values in arbitrary D-Brain phantom units.
```

**Standard (for D-Brain only)**:
```latex
\par\noindent\textit{Note:} $^\dagger$MD-MAE values are in arbitrary units from the D-Brain phantom intensity scale; interpret as relative comparisons across methods.
```

**Full (mixed D-Brain + Stanford)**:
```latex
\par\noindent\textit{Note:} $^\dagger$MD-MAE, AD-MAE, and RD-MAE for D-Brain are in arbitrary phantom intensity units and should be interpreted as relative comparisons across methods. Stanford HARDI diffusivity errors are in physical mm$^2$/s units.
```

---

## Next Steps

1. **Read** `paper/Sepulveda_dwmri_restormer.tex` carefully
2. **Identify** all tables with MD-MAE/AD-MAE/RD-MAE columns
3. **Remove** both TODO comments (lines 206, 447)
4. **Add** appropriate footnotes after each relevant `\end{table}`
5. **Compile** LaTeX to verify formatting
6. **Cross-check** with the registry values to ensure no accidental changes to numbers
