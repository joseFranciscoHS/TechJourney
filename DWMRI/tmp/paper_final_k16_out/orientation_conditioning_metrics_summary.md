# Orientation Conditioning Ablation: Metrics Summary

## Overview

This document summarizes the metrics for orientation conditioning experiments comparing:
- **Baseline RGS** (no gradient direction conditioning)
- **FiLM Conditioning** (target-only feature-space modulation)

The "orientation_encoding_OLD" jobs ran with the encoder inactive (116K params = baseline), so only Baseline vs FiLM comparison is valid.

**Authoritative source:** `registry.jsonl` lines 12–13 (baselines), 131–134 (FiLM).

---

## D-Brain Results (with Ground Truth)

### DRCNet (3D CNN)

| Metric | Baseline RGS | FiLM Conditioning | Change |
|--------|--------------|-------------------|--------|
| **n_params** | 116,002 | 120,546 | +4,544 (+3.92%) |
| **PSNR (full)** | 23.93 dB | **25.40 dB** | **+1.47 dB** |
| **SSIM (full)** | 0.4404 | **0.4552** | **+0.0148** |
| **MSE (full)** | 0.004048 | **0.002882** | **-28.8%** |
| **PSNR (ROI)** | 26.88 dB | 26.24 dB | **-0.64 dB** |
| **MSE (ROI)** | 0.002050 | 0.002374 | **+15.8%** |
| **FA MAE** | 0.2587 | **0.2405** | **-7.0%** |
| **MD MAE** | 3449 | 3452 | +0.1% |
| **AD MAE** | 8299 | 8310 | +0.1% |
| **RD MAE** | 1036 | 1040 | +0.4% |
| **Training time** | 2,091 s (34.9 min) | 7,569 s (126.2 min) | +3.6× |
| **Inference (per volume)** | 34.0 s | 35.3 s | +3.8% |

**Key observations:**
- FiLM improves full-volume PSNR by **+1.47 dB** but **degrades ROI PSNR by -0.64 dB**
- Full-volume MSE decreases by **28.8%**, but ROI MSE **increases by 15.8%** — global gain is not concentrated in brain tissue
- **7.0% improvement in FA error** — better diffusion tensor estimation in ROI
- Minimal inference overhead (~4%), training time increase due to longer convergence

---

### Restormer (3D Transformer)

| Metric | Baseline RGS | FiLM Conditioning | Change |
|--------|--------------|-------------------|--------|
| **n_params** | 177,883 | 184,699 | +6,816 (+3.83%) |
| **PSNR (full)** | 22.83 dB | **23.60 dB** | **+0.77 dB** |
| **SSIM (full)** | 0.4271 | **0.4380** | **+0.0109** |
| **MSE (full)** | 0.005206 | **0.004362** | **-16.2%** |
| **PSNR (ROI)** | 23.22 dB | 22.69 dB | **-0.53 dB** |
| **MSE (ROI)** | 0.003120 | 0.003522 | **+12.9%** |
| **FA MAE** | 0.2378 | 0.2603 | +9.5% (worse) |
| **MD MAE** | 3455 | **3423** | **-0.9%** |
| **AD MAE** | 8306 | 8241 | -0.8% |
| **RD MAE** | 1078 | 1034 | -4.1% |
| **Training time** | 7,626 s (127.1 min) | 19,859 s (331.0 min) | +2.6× |
| **Inference (per volume)** | 126.2 s | 130.3 s | +3.2% |

**Key observations:**
- FiLM improves full-volume PSNR by **+0.77 dB** but **degrades ROI PSNR by -0.53 dB**
- Full-volume MSE decreases by **16.2%**, but ROI MSE **increases by 12.9%**
- FA MAE is **worse** with FiLM (+9.5%), but MD/AD/RD errors improve slightly
- Restormer shows less full-volume benefit from FiLM than DRCNet
- Longer training time due to more FiLM layers (3 vs 2 in DRCNet)

---

## Stanford Results (no Ground Truth)

Stanford dataset has **no clean reference**, so only noisy-to-noisy metrics (PSNR/SSIM on self-comparisons) are available. DTI-derived maps (FA, MD) provide qualitative assessment.

**Important:** Stanford PSNR values are measured against the noisy input (no GT available). High PSNR indicates similarity to input, **NOT** denoising quality. Do not compare Stanford PSNR with D-Brain PSNR.

### DRCNet

| Metric | FiLM Conditioning |
|--------|-------------------|
| **n_params** | 120,546 |
| **PSNR (full)** | 38.27 dB |
| **SSIM (full)** | 0.9546 |
| **MSE (full)** | 0.0001489 |
| **PSNR (ROI)** | 45.06 dB |
| **MSE (ROI)** | 0.00005085 |
| **Training time** | 27,010 s (450.2 min / 7.5 hrs) |
| **Inference (per volume)** | 26.8 s |

### Restormer

| Metric | FiLM Conditioning |
|--------|-------------------|
| **n_params** | 184,699 |
| **PSNR (full)** | 25.17 dB |
| **SSIM (full)** | 0.6074 |
| **MSE (full)** | 0.003044 |
| **PSNR (ROI)** | 32.56 dB |
| **MSE (ROI)** | 0.0007836 |
| **Training time** | 23,590 s (393.2 min / 6.6 hrs) |
| **Inference (per volume)** | 85.4 s |

**Stanford observations:**
- No baseline RGS Stanford runs in the registry (can't compare)
- DRCNet shows much higher noisy-input PSNR than Restormer on Stanford (38 vs 25 dB) — reflects input preservation, not restoration quality
- Stanford has **G=150** gradients (vs G=60 on D-Brain), **b=2000 s/mm²** (vs b=2500)
- Inference time per volume is lower on Stanford (26.8s vs 35.3s for DRCNet) despite 2.5× more gradients — likely due to lower noise level (sigma=0.01 vs 0.1)

---

## Summary for Paper

### Quantitative comparison (D-Brain, sigma=0.1):

| Model | Conditioning | PSNR↑ | PSNR-ROI↑ | FA-MAE↓ | MD-MAE↓ | Params | Time/vol |
|-------|--------------|-------|-----------|---------|---------|--------|----------|
| DRCNet | None | 23.93 | **26.88** | 0.2587 | 3449 | 116K | 34.0s |
| DRCNet | **FiLM** | **25.40** | 26.24 | **0.2405** | 3452 | 120K | 35.3s |
| Restormer | None | 22.83 | 23.22 | **0.2378** | 3455 | 178K | 126.2s |
| Restormer | **FiLM** | **23.60** | 22.69 | 0.2603 | **3423** | 185K | 130.3s |

**Key takeaways:**
1. **FiLM improves full-volume PSNR** (+1.47 dB DRCNet, +0.77 dB Restormer) **but degrades ROI PSNR** (-0.64 dB DRCNet, -0.53 dB Restormer)
2. **ROI degradation is metric-dependent** — global smoothing may benefit full-volume PSNR while hurting brain-tissue fidelity
3. **FA error improves with FiLM in DRCNet** (-7.0%) but **worsens in Restormer** (+9.5%)
4. **MD/AD/RD errors** change little (DRCNet) or improve slightly (Restormer)
5. **Minimal inference overhead** (~4%), modest parameter increase (~4%)
6. **Training time increases** significantly (2.6–3.6×) due to longer convergence / more FiLM passes

### Recommendations for paper writeup:
- Position FiLM as an **exploratory** conditioning mechanism with **mixed results**, not uniformly beneficial
- Highlight **full-volume vs ROI tradeoff**: +1.47 dB full PSNR but -0.64 dB ROI PSNR for DRCNet
- Discuss **architecture dependency**: DRCNet gains more full-volume PSNR from explicit conditioning; Restormer shows smaller gains and worse FA
- **Investigate Restormer FA degradation** (possible overfitting or need for different FiLM placement/init)
- For **Stanford generalization**: report only FiLM results; note lack of baseline; clarify noisy-input PSNR is not restoration accuracy
- Suggested **figure**: FA/MD maps or residual panels for Stanford (see `pruebas_faltantes_todos_20260701.md` items #5–#6)

---

## Source Files

### D-Brain Baselines
- DRCNet: `tmp/paper_final_k16_out/drcnet_hybrid_rgs/metrics/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00045/`
- Restormer: `tmp/paper_final_k16_out/restormer_hybrid_rgs/metrics/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00045/`

### D-Brain FiLM
- DRCNet: `tmp/paper_final_k16_out/drcnet_hybrid_rgs/metrics/dbrain_film_conditioning/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00045/`
- Restormer: `tmp/paper_final_k16_out/restormer_hybrid_rgs/metrics/dbrain_film_conditioning/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00045/`

### Stanford FiLM
- DRCNet: `tmp/paper_final_k16_out/drcnet_hybrid_rgs/metrics/stanford_film_conditioning/b1000/rgs_G150_K16/noise_rician_sigma_0.01/learning_rate_0.00045/`
- Restormer: `tmp/paper_final_k16_out/restormer_hybrid_rgs/metrics/stanford_film_conditioning/b1000/rgs_G150_K16/noise_rician_sigma_0.01/learning_rate_0.00045/`

*Note: Stanford metric paths use `b1000` in directory naming (legacy config); actual acquisition is b=2000 s/mm² per DIPY.*

### Registry
- Main registry: `tmp/paper_final_k16_out/registry.jsonl`
- Job IDs: `drcnet_dbrain_rgs_final`, `restormer_dbrain_rgs_final`, `drcnet_dbrain_rgs_film_conditioning`, `restormer_dbrain_rgs_film_conditioning`, `drcnet_stanford_rgs_film_conditioning`, `restormer_stanford_rgs_film_conditioning`
