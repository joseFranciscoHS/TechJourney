# Orientation Conditioning Ablation: Metrics Summary

## Overview

This document summarizes the metrics for orientation conditioning experiments comparing:
- **Baseline RGS** (no gradient direction conditioning)
- **FiLM Conditioning** (target-only feature-space modulation)

The "orientation_encoding_OLD" jobs ran with the encoder inactive (116K params = baseline), so only Baseline vs FiLM comparison is valid.

---

## D-Brain Results (with Ground Truth)

### DRCNet (3D CNN)

| Metric | Baseline RGS | FiLM Conditioning | Improvement |
|--------|--------------|-------------------|-------------|
| **n_params** | 116,002 | 120,546 | +4,544 (+3.92%) |
| **PSNR (full)** | 23.93 dB | **25.40 dB** | **+1.47 dB** |
| **SSIM (full)** | 0.4404 | **0.4552** | **+0.0148** |
| **MSE (full)** | 0.004048 | **0.002882** | **-28.8%** |
| **PSNR (ROI)** | 28.48 dB | **30.40 dB** | **+1.92 dB** |
| **MSE (ROI)** | 0.002050 | **0.001266** | **-38.2%** |
| **FA MAE** | 0.2606 | **0.2405** | **-7.7%** |
| **MD MAE** | 3539 | **3452** | **-2.5%** |
| **AD MAE** | 8511 | **8310** | **-2.4%** |
| **RD MAE** | 1069 | **1040** | **-2.7%** |
| **Training time** | 2,091 s (34.9 min) | 7,569 s (126.2 min) | +3.6× |
| **Inference (per volume)** | 34.0 s | 35.3 s | +3.8% |

**Key observations:**
- FiLM improves PSNR by **1.47 dB** overall, **1.92 dB** in ROI
- **38% reduction in ROI MSE** - substantial improvement in brain regions
- **7.7% improvement in FA error** - better diffusion tensor estimation
- Minimal inference overhead (~4%), training time increase due to longer convergence

---

### Restormer (3D Transformer)

| Metric | Baseline RGS | FiLM Conditioning | Improvement |
|--------|--------------|-------------------|-------------|
| **n_params** | 177,883 | 184,699 | +6,816 (+3.83%) |
| **PSNR (full)** | 22.83 dB | **23.60 dB** | **+0.77 dB** |
| **SSIM (full)** | 0.4271 | **0.4380** | **+0.0109** |
| **MSE (full)** | 0.005206 | **0.004362** | **-16.2%** |
| **PSNR (ROI)** | 26.56 dB | **27.94 dB** | **+1.38 dB** |
| **MSE (ROI)** | 0.003120 | **0.002068** | **-33.7%** |
| **FA MAE** | 0.2424 | **0.2603** | -7.4% (worse) |
| **MD MAE** | 3502 | **3423** | **+2.3%** |
| **AD MAE** | 8420 | **8241** | **+2.1%** |
| **RD MAE** | 1080 | **1034** | **+4.3%** |
| **Training time** | 7,626 s (127.1 min) | 19,859 s (331.0 min) | +2.6× |
| **Inference (per volume)** | 126.2 s | 130.3 s | +3.2% |

**Key observations:**
- FiLM improves PSNR by **0.77 dB** overall, **1.38 dB** in ROI
- **33.7% reduction in ROI MSE**
- FA MAE is **worse** with FiLM (+7.4%), but MD/AD/RD errors improve
- Restormer shows less benefit from FiLM than DRCNet (possible overfitting or initialization issue)
- Longer training time due to more FiLM layers (3 vs 2 in DRCNet)

---

## Stanford Results (no Ground Truth)

Stanford dataset has **no clean reference**, so only noisy-to-noisy metrics (PSNR/SSIM on self-comparisons) are available. DTI-derived maps (FA, MD) provide qualitative assessment.

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
- DRCNet shows much higher PSNR than Restormer on Stanford (38 vs 25 dB)
- Stanford has **G=150** gradients (vs G=60 on D-Brain), **b=1000** (vs b=2500)
- Inference time per volume is lower on Stanford (26.8s vs 35.3s for DRCNet) despite 2.5× more gradients - likely due to lower noise level (sigma=0.01 vs 0.1)

---

## Summary for Paper

### Quantitative comparison (D-Brain, sigma=0.1):

| Model | Conditioning | PSNR↑ | PSNR-ROI↑ | FA-MAE↓ | MD-MAE↓ | Params | Time/vol |
|-------|--------------|-------|-----------|---------|---------|--------|----------|
| DRCNet | None | 23.93 | 28.48 | 0.2606 | 3539 | 116K | 34.0s |
| DRCNet | **FiLM** | **25.40** | **30.40** | **0.2405** | **3452** | 120K | 35.3s |
| Restormer | None | 22.83 | 26.56 | **0.2424** | 3502 | 178K | 126.2s |
| Restormer | **FiLM** | **23.60** | **27.94** | 0.2603 | **3423** | 185K | 130.3s |

**Key takeaways:**
1. **FiLM consistently improves PSNR** (+0.77 to +1.47 dB), especially in ROI (+1.38 to +1.92 dB)
2. **DRCNet benefits more from FiLM than Restormer** (1.47 dB vs 0.77 dB improvement)
3. **FA error improves with FiLM in DRCNet** (-7.7%) but **worsens in Restormer** (+7.4%)
4. **MD/AD/RD errors improve** in both architectures with FiLM
5. **Minimal inference overhead** (~4%), modest parameter increase (~4%)
6. **Training time increases** significantly (2.6-3.6×) due to longer convergence / more FiLM passes

### Recommendations for paper writeup:
- Position FiLM as a **low-overhead, high-benefit** conditioning mechanism
- Highlight **ROI improvements** (38% MSE reduction in DRCNet) as clinically relevant
- Discuss **architecture dependency**: DRCNet (simpler inductive bias) gains more from explicit conditioning than Restormer (richer self-attention)
- **Investigate Restormer FA degradation** (possible overfitting or need for different FiLM placement/init)
- For **Stanford generalization**: report only FiLM results, note lack of baseline for comparison
- Suggested **figure**: Side-by-side FA/MD maps (Baseline, FiLM, GT) for D-Brain, visually showing ROI improvement

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

### Registry
- Main registry: `tmp/paper_final_k16_out/registry.jsonl`
- Job IDs: `drcnet_dbrain_rgs_final`, `restormer_dbrain_rgs_final`, `drcnet_dbrain_rgs_film_conditioning`, `restormer_dbrain_rgs_film_conditioning`, `drcnet_stanford_rgs_film_conditioning`, `restormer_stanford_rgs_film_conditioning`
