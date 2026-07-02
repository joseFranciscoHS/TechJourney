# 020 — K=16 Canonical Rerun Results Update

## Context for the LLM

You are updating an existing LaTeX manuscript on **Hybrid RGS** (Random Gradient Subset) self-supervised DW-MRI denoising. The paper [`paper/Sepulveda_dwmri_restormer.tex`](../../paper/Sepulveda_dwmri_restormer.tex) currently cites canonical K=16 baseline metrics from May 2026 runs in `tmp/paper_final_k16_out/registry.jsonl`.

On **June 28, 2026**, the four canonical jobs were re-executed via [`experiments/rerun_k16_base.sh`](../../experiments/rerun_k16_base.sh) with `exp_id=paper_final_k16_rerun`. Outputs live in `tmp/paper_final_k16_rerun_20260628T042410Z/`. This rerun:

- Produces a **clean registry** without the `inference_time_grid` metric corruption documented in [`017_registry_conflict_documentation.md`](017_registry_conflict_documentation.md).
- Uses the same hyperparameters as [`experiments/paper_manifest_final.yaml`](../../experiments/paper_manifest_final.yaml): D-Brain `seed=91021`, `reproducible=true`, `subset_fraction=0.6`; Stanford `seed=91022`, `reproducible=true`.
- Replaces metrics **only** for jobs `drcnet_dbrain_rgs_final`, `restormer_dbrain_rgs_final`, `drcnet_stanford_rgs_final`, `restormer_stanford_rgs_final`.

**FiLM-conditioned runs were NOT re-executed.** Keep FiLM absolute values from the old registry (`tmp/paper_final_k16_out/registry.jsonl`, lines 131–132) but **recompute baseline rows, Δ rows, and narrative** using the new rerun baselines.

**Scope decision (confirmed):** Synchronize **all** paper locations that cite the canonical `*_rgs_final` K=16 jobs — main comparison, ablation tables with K=16 RGS rows, sigma σ=0.10 row, 3D rows, objective-controlled Hybrid RGS row, FiLM baseline, Stanford K=16 row, registry provenance, and implementation-details seeds.

**Do NOT change:** MP-PCA / Patch2Self / MD-S2S / supervised rows; sequential ablation rows; K-sweep K≠16; sigma≠0.10; 2D ablation rows; FiLM conditioned absolute values; `\TODO{}` figure placeholders; abstract (no numeric claims).

---

## Files to attach

Attach these files when you send this prompt. Paths are relative to `DWMRI/`.

| File | Purpose |
|------|---------|
| `paper/Sepulveda_dwmri_restormer.tex` | Current manuscript to patch in place |
| `tmp/paper_final_k16_rerun_20260628T042410Z/registry.jsonl` | **Authoritative source** for rerun baselines (use lines 3–6 — rows with complete `quality_metrics_*`) |
| `tmp/paper_final_k16_rerun_20260628T042410Z/paper_tables/paper_metrics_summary.csv` | Flat summary of the 4 rerun jobs |
| `tmp/paper_final_k16_rerun_20260628T042410Z/paper_tables/registry_summary.csv` | Runtime and parameter counts |
| `experiments/rerun_k16_base.sh` | Exact commands and config overrides for provenance |
| `tmp/paper_final_k16_out/registry.jsonl` | FiLM conditioned values only (lines 131–132: `*_film_conditioning` jobs) |
| `tmp/paper_final_k16_out/writing/017_registry_conflict_documentation.md` | Background on why the rerun supersedes May-2026 baselines |

---

## Authoritative values (rerun, rounded for paper)

### Rounding convention

| Quantity | Decimals |
|----------|----------|
| PSNR (dB) | 2 |
| SSIM | 4 |
| MSE | 5 in main table; compute % changes from full precision |
| FA-MAE | 4 |
| MD-MAE | integer |
| Time/vol (s) | 1 |
| Params (M) | 3 (unchanged: DRCNet 0.116, Restormer 0.178) |

### D-Brain σ=0.1, K=16 — canonical baselines (rerun registry lines 3–4)

| Metric | DRCNet-Hybrid-RGS | Restormer-Hybrid-RGS |
|--------|-------------------|----------------------|
| PSNR (full) | **23.90** | **22.98** |
| SSIM (full) | **0.4402** | **0.4288** |
| MSE (full) | **0.00408** | **0.00504** |
| PSNR-ROI | **26.93** | **23.43** |
| SSIM-ROI | **0.5247** | **0.5107** |
| MSE-ROI | **0.00203** | **0.00454** |
| FA-MAE | **0.2575** | **0.2238** |
| MD-MAE | **3451** | **3440** |
| Time/vol (s) | **34.1** | **126.8** |
| Params | 116K (0.116M) | 178K (0.178M) |

### Stanford K=16 — noisy-input PSNR-ROI (rerun registry lines 5–6)

| Architecture | Noisy-input PSNR-ROI | Time/vol (s) |
|--------------|----------------------|--------------|
| DRCNet | **38.07** | **25.2** |
| Restormer | **26.05** | **83.1** |

*Interpretation unchanged: measured against acquired noisy image, not ground truth.*

### FiLM table — baseline update + unchanged FiLM rows (old registry lines 131–132)

**Keep FiLM absolute values:**

| Model | Conditioning | PSNR | PSNR-ROI | FA-MAE | MD-MAE | Params | Time/vol |
|-------|--------------|------|----------|--------|--------|--------|----------|
| DRCNet | FiLM | 25.40 | 26.24 | 0.2405 | 3452 | 120K | 35.3 s |
| Restormer | FiLM | 23.60 | 22.69 | 0.2603 | 3423 | 185K | 130.3 s |

**Update baseline rows to rerun values** (see table above).

**Recompute Δ rows:**

| Model | Δ PSNR | Δ PSNR-ROI | Δ FA-MAE | Δ MD-MAE | Δ Params | Δ Time |
|-------|--------|------------|----------|----------|----------|--------|
| DRCNet | **+1.51** | **−0.69** | **−6.6%** | +0.0% | +3.9% | +3.2% |
| Restormer | **+0.63** | **−0.74** | **+16.4%** | −0.5% | +3.8% | +2.8% |

**Recompute MSE narrative (DRCNet):** full-volume MSE **−29.3%**; ROI MSE **+17.2%**.

**Recompute MSE narrative (Restormer):** full-volume MSE **−13.4%**; ROI MSE **+18.4%**.

---

## Old → new mapping (paper currently has mixed sources)

Use this table to find and replace stale numbers. After editing, **one canonical set** (rerun values above) must appear everywhere that cites `*_rgs_final` K=16.

### D-Brain canonical baselines

| Location / metric | Paper now (stale) | Rerun (use) |
|-------------------|-------------------|-------------|
| DRCNet PSNR (full) | 23.82 or 23.93 | **23.90** |
| DRCNet PSNR-ROI | 26.90 or 26.88 | **26.93** |
| DRCNet SSIM-ROI | 0.5240 | **0.5247** |
| DRCNet FA-MAE | 0.2499 or 0.2587 | **0.2575** |
| DRCNet MD-MAE | 3458 or 3449 | **3451** |
| DRCNet MSE-ROI | 0.00204 | **0.00203** |
| DRCNet Time/vol | 35.2 or 34.0 | **34.1** |
| Restormer PSNR (full) | 23.00 or 22.83 | **22.98** |
| Restormer PSNR-ROI | 23.39 or 23.22 | **23.43** |
| Restormer SSIM-ROI | 0.5116 | **0.5107** |
| Restormer FA-MAE | 0.2577 or 0.2378 | **0.2238** |
| Restormer MD-MAE | 3438 or 3455 | **3440** |
| Restormer MSE-ROI | 0.00458 | **0.00454** |
| Restormer Time/vol | 129.9 or 126.2 | **126.8** |

### Stanford K=16

| Metric | Paper now | Rerun |
|--------|-----------|-------|
| DRCNet noisy PSNR-ROI | 38.06 | **38.07** |
| DRCNet time | 25.5 s | **25.2 s** |
| Restormer noisy PSNR-ROI | 26.12 | **26.05** |
| Restormer time | 84.8 s | **83.1 s** |

### Narrative deltas to recalculate (Main Comparison §4.1)

Using Patch2Self PSNR-ROI = 21.31 dB and MSE-ROI = 0.00739 (unchanged baseline row):

- DRCNet vs Patch2Self PSNR-ROI gain: **5.62 dB** (was 5.59)
- DRCNet vs MD-S2S PSNR-ROI gain: **11.65 dB** (was 11.62)
- DRCNet ROI MSE reduction vs Patch2Self: **~73%** (was ~72%)

**Important narrative change:** Restormer FA-MAE (**0.2238**) is now **lower** (better) than DRCNet (**0.2575**), while DRCNet still leads on PSNR-ROI. Revise the Restormer paragraph in Main Comparison accordingly — do not claim Restormer has higher FA-MAE.

---

## Prompt

Patch [`paper/Sepulveda_dwmri_restormer.tex`](../../paper/Sepulveda_dwmri_restormer.tex) in place. Produce **minimal diffs** — update numbers, recalculate derived claims, and fix provenance; do not rewrite unrelated prose.

### 1. Registry provenance (`\paragraph{Registry provenance}`)

Replace the May-16-2026 reference with:

- `exp_id`: `paper_final_k16_rerun`
- Completion date: **June 28, 2026**
- Registry path: `tmp/paper_final_k16_rerun_20260628T042410Z/registry.jsonl`
- Job IDs unchanged: `drcnet_dbrain_rgs_final`, `restormer_dbrain_rgs_final`, `drcnet_stanford_rgs_final`, `restormer_stanford_rgs_final`
- Note that D-Brain metrics use `best_loss_checkpoint` with Monte Carlo inference `N_c=16`, `N_p=12` (D-Brain) / `N_p=23` (Stanford)

### 2. Implementation Details — Training paragraph (~line 377)

Fix the seed / reproducibility text to match the rerun:

- **Remove** "seed 42" and `reproducible: false`
- **State:** D-Brain training uses `seed=91021`, Stanford uses `seed=91022`, both with `reproducible: true`
- **Add:** D-Brain uses `subset_fraction=0.6` (random patch subsampling for training speed)
- Keep the caveat that cuDNN nondeterminism may still prevent bitwise GPU reproducibility

### 3. Tables and inline text — update in scope

| Label | What to update |
|-------|----------------|
| `tab:main_comparison` | DRCNet-Hybrid-RGS and Restormer-Hybrid-RGS rows (all metric columns + time) |
| Main Comparison narrative (§4.1) | All inline numbers for proposed methods; recalculate vs-Patch2Self / vs-MD-S2S claims; fix Restormer FA-MAE comparison |
| `tab:film_ablation` | Baseline rows + Δ rows; keep FiLM absolute rows |
| FiLM narrative (§3.x) | Baseline PSNR values, Δ dB, MSE % changes, ROI degradation magnitudes (−0.69 / −0.74 dB) |
| `tab:sigma_sweep` | σ=0.10 row only: DRCNet and Restormer PSNR-ROI and FA-MAE in parentheses |
| `tab:sampling_ablations` | Both K=16 RGS rows (sequential-vs-RGS block and K-sweep block) + inline mentions of 26.90→26.93, 23.39→23.43, times |
| `tab:objective_controlled_ablation` | DRCNet Hybrid RGS row + opening paragraph citing 26.90 dB |
| `tab:3d_vs_2d` | 3D DRCNet and 3D Restormer rows + narrative timing (34.1 s / 126.8 s) |
| `tab:stanford_k_sweep` | K=16 rows for DRCNet and Restormer |
| Stanford narrative | Any inline K=16 noisy-input PSNR-ROI or timing for canonical (non-FiLM) runs |

### 4. Out of scope — do not edit

- Baseline rows (MP-PCA, Patch2Self, MD-S2S, supervised)
- Sequential sampling rows, K∈{5,10,24,30}, sigma≠0.10, 2D ablation rows
- FiLM conditioned absolute metrics (25.40, 26.24, etc.) — only recompute deltas
- `\TODO{}` blocks at lines ~453, 465, 495
- Abstract, Discussion, Conclusion (unless they contain specific stale numbers — grep first; currently they do not)

### 5. Consistency rule

After editing, grep the `.tex` for `23.82`, `23.00`, `26.90`, `23.39`, `0.2499`, `0.2577`, `26.88`, `23.22`, `May 16, 2026`. None should remain except inside comments or historical context you explicitly preserve.

### 6. Tone

- Keep academic, cautious tone.
- Do not oversell small metric shifts (<0.1 dB).
- Acknowledge Restormer's improved FA-MAE without reversing the conclusion that DRCNet is the stronger image-domain / practical choice.

---

## Expected output

Hand back:

1. **Patched LaTeX** — either the full updated `.tex` or a numbered list of search/replace hunks with enough context for unique matching.
2. **Verification table** (markdown):

| Metric | Old (paper) | New (rerun) | Updated in |
|--------|-------------|-------------|------------|

   One row per changed value, with `\ref{...}` or line number.

3. **Post-update grep checklist** (report pass/fail):

```bash
# Stale canonical values should be gone
grep -n '23\.82\|23\.00\|26\.90\|23\.39\|0\.2499\|0\.2577\|May 16, 2026' \
  paper/Sepulveda_dwmri_restormer.tex

# New canonical values should appear
grep -n '26\.93\|23\.43\|0\.2238\|June 28, 2026\|paper_final_k16_rerun' \
  paper/Sepulveda_dwmri_restormer.tex

# FiLM ROI deltas updated
grep -n '\-0\.69\|\-0\.74' paper/Sepulveda_dwmri_restormer.tex

# Baselines untouched
grep -n 'Patch2Self.*21\.31\|MP-PCA.*28\.44' paper/Sepulveda_dwmri_restormer.tex
```

4. **Brief summary** (3–5 bullets) of narrative changes, especially Restormer FA-MAE and FiLM Δ magnitudes.

---

## Quick reference — raw registry excerpts

**Rerun** (`tmp/paper_final_k16_rerun_20260628T042410Z/registry.jsonl`):

- Line 3: `drcnet_dbrain_rgs_final` — PSNR 23.898, PSNR-ROI 26.931, FA 0.2575, MD 3450.68, time 34.15 s
- Line 4: `restormer_dbrain_rgs_final` — PSNR 22.976, PSNR-ROI 23.428, FA 0.2238, MD 3440.45, time 126.76 s
- Line 5: `drcnet_stanford_rgs_final` — noisy PSNR-ROI 38.069, time 25.20 s
- Line 6: `restormer_stanford_rgs_final` — noisy PSNR-ROI 26.049, time 83.06 s

**FiLM (unchanged)** (`tmp/paper_final_k16_out/registry.jsonl`):

- Line 131: `drcnet_dbrain_rgs_film_conditioning`
- Line 132: `restormer_dbrain_rgs_film_conditioning`
