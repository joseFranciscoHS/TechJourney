# 021 — Sigma Sweep Baseline Backfill (Patch2Self & MD-S2S at σ=0.15, 0.20)

## Context for the LLM

You are updating an existing LaTeX manuscript on **Hybrid RGS** self-supervised DW-MRI denoising. The sigma-robustness subsection (`\subsection{Robustness to Noise Level}`) and **Figure 2** (`fig:sigma_robustness`) currently show `--` for **Patch2Self** and **MD-S2S** at $\sigma \in \{0.15, 0.20\}$ in `tab:sigma_sweep`, with a caption stating that missing baseline points are omitted when runs are unavailable.

**This is misleading.** The runs **were executed successfully** on May 29, 2026 according to `tmp/paper_final_k16_out/driver_state.json`:

| Job ID | Status | σ |
|--------|--------|---|
| `p2s_dbrain_dipy_sigma_150_final` | succeeded | 0.15 |
| `p2s_dbrain_dipy_sigma_200_final` | succeeded | 0.20 |
| `mds2s_dbrain_sigma_150_final` | succeeded | 0.15 |
| `mds2s_dbrain_sigma_200_final` | succeeded | 0.20 |

The gap is **artifact staging**, not missing experiments:

- MP-PCA metrics for all four σ levels are present under `tmp/paper_final_k16_out/baselines/mppca/dbrain_sigma_{050,100,150,200}/`.
- Patch2Self and MD-S2S metrics for σ=0.05 and σ=0.10 appear in `paper_metrics_summary.csv`.
- Patch2Self and MD-S2S metrics for σ=0.15 and σ=0.20 were computed (see run logs) but **`metrics.json` / `metrics_roi.json` were never copied into `paper_final_k16_out/`**, so `collect_paper_artifacts.py` and the figure/table pipeline could not ingest them.
- Baseline jobs use `append_registry_flags: false` in the manifest; they are **not** in `registry.jsonl` by design.

**Scope decision:** Backfill σ=0.15 and σ=0.20 **Patch2Self (DIPY OLS)** and **MD-S2S** only. Do **not** change MP-PCA, proposed-method σ rows (already updated via prompt 020 for σ=0.10 canonical rerun), σ=0.05/0.10 baseline values, or unrelated sections.

---

## Files to attach

Attach these files when you send this prompt. Paths are relative to `DWMRI/`.

| File | Purpose |
|------|---------|
| `paper/Sepulveda_dwmri_restormer.tex` | Manuscript to patch in place |
| `tmp/paper_final_k16_out/driver_state.json` | Proof jobs succeeded (`p2s_dbrain_dipy_sigma_150_final`, etc.) |
| `tmp/paper_final_k16_out/runs/p2s_dbrain_dipy_sigma_150_final.log` | Authoritative Patch2Self σ=0.15 metrics |
| `tmp/paper_final_k16_out/runs/p2s_dbrain_dipy_sigma_200_final.log` | Authoritative Patch2Self σ=0.20 metrics |
| `tmp/paper_final_k16_out/runs/mds2s_dbrain_sigma_150_final.log` | Authoritative MD-S2S σ=0.15 metrics |
| `tmp/paper_final_k16_out/runs/mds2s_dbrain_sigma_200_final.log` | Authoritative MD-S2S σ=0.20 metrics |
| `tmp/paper_final_k16_out/paper_tables/paper_metrics_summary.csv` | Current flat summary (verify σ=0.15/0.20 rows after backfill) |
| `experiments/collect_paper_artifacts.py` | Regenerates `paper_metrics_summary.csv` from staged `metrics.json` trees |
| `experiments/paper_manifest_final.yaml` | Job definitions (`p2s_dbrain_dipy_sigma_150_final`, `mds2s_dbrain_sigma_150_final`, etc.) |
| `tmp/paper_final_k16_out/writing/005_sigma_robustness.md` | Original drafting intent for this subsection |
| `paper/figures/sigma_robustness_psnr_roi.png` | Figure to regenerate after table is complete |

---

## Authoritative values (from run logs, rounded for paper)

### Rounding convention

| Quantity | Decimals |
|----------|----------|
| PSNR-ROI (dB) | 2 |

*Patch2Self and MD-S2S rows in `tab:sigma_sweep` report PSNR-ROI only (no FA-MAE parentheses), matching σ=0.05 and σ=0.10.*

### Patch2Self (DIPY OLS backend)

| σ | PSNR (full) | PSNR-ROI | SSIM-ROI | Source log line |
|---|-------------|----------|----------|-----------------|
| 0.15 | 17.91 | **21.48** | 0.4446 | `p2s_dbrain_dipy_sigma_150_final.log` |
| 0.20 | 15.94 | **19.10** | 0.4026 | `p2s_dbrain_dipy_sigma_200_final.log` |

### MD-S2S (Conv2D, G=60 volumes)

| σ | PSNR (full) | PSNR-ROI | SSIM-ROI | Source log line |
|---|-------------|----------|----------|-----------------|
| 0.15 | 14.84 | **15.10** | 0.4694 | `mds2s_dbrain_sigma_150_final.log` |
| 0.20 | 14.08 | **14.57** | 0.4383 | `mds2s_dbrain_sigma_200_final.log` |

### Full sigma sweep after backfill (PSNR-ROI only, for figure QA)

| σ | MP-PCA | Patch2Self | MD-S2S | DRCNet-Hybrid-RGS | Restormer-Hybrid-RGS |
|---|--------|------------|--------|-------------------|----------------------|
| 0.05 | 33.33 | 23.74 | 16.77 | 29.76 | 27.19 |
| 0.10 | 28.44 | 21.31 | 15.28 | 26.93 | 23.43 |
| 0.15 | 25.54 | **21.48** | **15.10** | 25.11 | 24.63 |
| 0.20 | 23.55 | **19.10** | **14.57** | 23.52 | 23.02 |

**Narrative implications (do not oversell):**

- At σ=0.15 and σ=0.20, DRCNet-Hybrid-RGS remains well above Patch2Self (+3.6 dB at σ=0.15; +4.4 dB at σ=0.20) and MD-S2S (+10.0 dB / +9.0 dB).
- Patch2Self degrades monotonically with σ (23.74 → 21.31 → 21.48 → 19.10). The σ=0.15 value is slightly higher than σ=0.10 in ROI PSNR; treat as run-to-run / pipeline variance, not a claim of improvement at higher noise.
- MD-S2S remains the weakest image-domain baseline at high σ, as at σ=0.10.
- MP-PCA still leads PSNR-ROI at σ=0.15 and σ=0.20; do not reverse that conclusion.

**Provenance note:** MD-S2S logs at σ=0.15/0.20 report ROI mask coverage ~61.1% vs ~31.9% for Patch2Self on the same phantom. Both use threshold 0.02 but may differ in mask construction. Do not speculate in the paper; just use the logged ROI metrics consistently with prior σ rows.

---

## Prompt

Complete the sigma-sweep baseline backfill and sync the paper. Work in two phases: **(A) artifact staging**, then **(B) manuscript + figure**.

### Phase A — Stage metrics artifacts (do this first)

Create `metrics.json` and `metrics_roi.json` under `tmp/paper_final_k16_out/` using paths consistent with existing `paper_metrics_summary.csv` rows for σ=0.05/0.10:

```
tmp/paper_final_k16_out/p2s/metrics/dbrain/bvalue_2500/noise_sigma_0.15/backend_dipy_model_ols/
tmp/paper_final_k16_out/p2s/metrics/dbrain/bvalue_2500/noise_sigma_0.2/backend_dipy_model_ols/
tmp/paper_final_k16_out/mds2s/metrics/dbrain/bvalue_2500/num_volumes_60/noise_sigma_0.15/learning_rate_0.0001/
tmp/paper_final_k16_out/mds2s/metrics/dbrain/bvalue_2500/num_volumes_60/noise_sigma_0.2/learning_rate_0.0001/
```

Populate JSON from the authoritative log excerpts above. Use keys `psnr`, `ssim`, `mse` in `metrics.json` and `psnr`, `ssim`, `mse` in `metrics_roi.json` (matching existing collectors).

Then regenerate the summary:

```bash
python experiments/collect_paper_artifacts.py \
  --output-root tmp/paper_final_k16_out \
  --registry tmp/paper_final_k16_out/registry.jsonl \
  --out-dir tmp/paper_final_k16_out/paper_tables
```

Verify `paper_metrics_summary.csv` now contains four new rows (`patch2self` and `mds2s` at σ=0.15 and σ=0.20).

**Alternative:** If artifact files cannot be recovered, re-run only the four manifest jobs (`p2s_dbrain_dipy_sigma_150_final`, etc.) with outputs directed into `paper_final_k16_out/`. Prefer log-based staging when logs are intact.

### Phase B — Patch `paper/Sepulveda_dwmri_restormer.tex`

Produce **minimal diffs**. Update only sigma-sweep-related prose, table, caption, and figure asset.

#### 1. Table `tab:sigma_sweep` (σ=0.15 and σ=0.20 rows only)

Replace `--` with rounded PSNR-ROI:

| Row | Patch2Self | MD-S2S |
|-----|------------|--------|
| σ=0.15 | **21.48** | **15.10** |
| σ=0.20 | **19.10** | **14.57** |

Leave MP-PCA and proposed-method columns unchanged.

#### 2. Setup paragraph (~line 297)

Replace language implying baselines are unavailable at high σ, e.g.:

- **Remove:** “Patch2Self and MD-S2S are available for the lower-noise settings in the current registry and are included where present.”
- **Replace with:** Patch2Self (DIPY OLS) and MD-S2S are evaluated at all four noise levels; each σ uses independently corrupted D-Brain data with `seed=91021`.

#### 3. Results narrative (§3.x Robustness to Noise Level)

Add **one short sentence** after the monotonic-degradation paragraph, noting that Patch2Self and MD-S2S also degrade at σ=0.15 and σ=0.20 (e.g. Patch2Self 21.48 dB and 19.10 dB; MD-S2S 15.10 dB and 14.57 dB PSNR-ROI), and that Hybrid RGS retains a substantial margin over both at high noise. Keep the existing MP-PCA narrative (MP-PCA leads at high σ). Do **not** rewrite the full subsection.

Optional: in the σ=0.05 paragraph, no change needed. Do **not** add new FA-MAE claims for Patch2Self/MD-S2S at high σ unless `dti_metrics.json` is staged.

#### 4. Table caption `tab:sigma_sweep`

Update to state that all five methods are reported at all four σ levels (remove “included where available in the current registry”).

#### 5. Figure `fig:sigma_robustness`

Regenerate `paper/figures/sigma_robustness_psnr_roi.png` so Patch2Self and MD-S2S curves include **four points** each (not truncated at σ=0.10). Use the full table above.

Update figure caption:

- **Remove:** “with missing baseline points omitted where runs are unavailable”
- **Replace with:** all five methods evaluated at each σ; Patch2Self uses DIPY OLS; each method trained/evaluated at the same noise level.

Match existing figure style (methods, colors, labels) if a plotting script exists; otherwise reproduce from the authoritative table.

### Out of scope — do not edit

- `tab:main_comparison` Patch2Self / MD-S2S rows (σ=0.10 only)
- MP-PCA values at any σ
- DRCNet / Restormer σ-sweep rows (already current)
- Abstract, Discussion, Conclusion
- `\TODO{}` placeholders elsewhere
- `p2s_dbrain_sklearn_reference_sigma_*` jobs (pending; not in manifest for paper reporting)

---

## Expected output

Hand back:

1. **Staged artifacts** — four `metrics.json` + `metrics_roi.json` pairs (or confirmation of re-run), plus regenerated `paper_metrics_summary.csv`.
2. **Patched LaTeX** — minimal hunks for `tab:sigma_sweep`, setup/results paragraphs, table caption, figure caption.
3. **Regenerated figure** — `paper/figures/sigma_robustness_psnr_roi.png` with complete curves.
4. **Verification table:**

| Metric | Old (paper) | New (backfill) | Updated in |
|--------|-------------|----------------|------------|
| P2S PSNR-ROI σ=0.15 | `--` | 21.48 | `tab:sigma_sweep` |
| … | … | … | … |

5. **Post-update grep checklist:**

```bash
# Placeholders should be gone from sigma table
grep -n '0\.15.*--\|0\.20.*--' paper/Sepulveda_dwmri_restormer.tex

# New values present
grep -n '21\.48\|19\.10\|15\.10\|14\.57' paper/Sepulveda_dwmri_restormer.tex

# Stale "unavailable" language should be gone
grep -n 'unavailable\|where present\|where available' paper/Sepulveda_dwmri_restormer.tex
```

6. **Brief summary** (3–5 bullets) of what changed and confirmation that runs were restaged rather than re-invented.

---

## Quick reference — raw log excerpts

**Patch2Self σ=0.15** (`p2s_dbrain_dipy_sigma_150_final.log`):
- Full: PSNR 17.914, SSIM 0.3630, MSE 0.01616
- ROI: PSNR 21.480, SSIM 0.4446, MSE 0.00711

**Patch2Self σ=0.20** (`p2s_dbrain_dipy_sigma_200_final.log`):
- Full: PSNR 15.941, SSIM 0.3280, MSE 0.02546
- ROI: PSNR 19.096, SSIM 0.4026, MSE 0.01231

**MD-S2S σ=0.15** (`mds2s_dbrain_sigma_150_final.log`):
- Full: PSNR 14.839, SSIM 0.3887, MSE 0.03282
- ROI: PSNR 15.096, SSIM 0.4694, MSE 0.03093

**MD-S2S σ=0.20** (`mds2s_dbrain_sigma_200_final.log`):
- Full: PSNR 14.077, SSIM 0.3622, MSE 0.03911
- ROI: PSNR 14.567, SSIM 0.4383, MSE 0.03494
