---
name: j-invariance-dwmri-reviewer
description: >-
  Peer review and Q&A for J-invariant and blind-spot self-supervised denoising of
  diffusion-weighted MRI (DWMRI): Noise2Void-style masking, Patch2Self angular
  redundancy, MD-S2S / Scheme 2 hybrid training, masked MSE, assumptions and
  threats to validity, and sample-efficient training. Use when the user asks
  about J-invariance, peer review of methods, mathematical justification, whether
  a pipeline is J-invariant, low training data, self-supervised DWI denoising, or
  when working in src/discussion or J_invariance_DWMRI_denoising_report.md.
---

# J-invariance and DWMRI denoising — reviewer skill

## When to use

Apply this skill when the user:

- Asks whether a procedure is **J-invariant** or requests a **peer review** of methods text or code.
- Wants help with **notation, assumptions, or bias** in self-supervised denoising.
- Asks how to denoise DWMRI with **little labeled data** or **limited training data** (favor self-supervised and inductive-bias levers).

## Peer-review template

Structure answers (or document reviews) with:

1. **Claim** — What exactly is claimed (e.g. “unbiased risk minimization”, “J-invariant training”)?
2. **Strengths** — What is sound or well aligned with the literature?
3. **Assumptions** — Noise model (independence, homoscedasticity, Rician vs Gaussian); signal correlation across space and across gradient directions; what is conditioned on.
4. **Limitations / threats** — Leakage (normalization, batch stats, overlapping patches), correlated artifacts, motion, g-factor noise, partial volume, implementation details (zero vs random fill at masks).
5. **Falsification** — What experiment or ablation would challenge the claim?

## J-invariance checklist

Work through explicitly:

1. **Measurement vector** — What is \(x\)? (e.g. stacked DWI volumes on a patch, single-channel target.)
2. **Index set \(J\)** — Which coordinates are supervised at this step? Fixed or random per forward pass?
3. **Predictor map \(f\)** — Show that **\(f(x)_J\)** does not depend on the **noisy** values **\(x_J\)** (inputs at \(J\) must not carry those observations; common pattern: multiply by mask so occluded sites are zero or constant).
4. **Loss** — Training loss uses noisy targets only at \(J\) (or the appropriate blind-spot sites); confirm it matches the implemented mask convention (`1 - mask` vs `mask` in code).
5. **Ideal theory link** — Under stated independence of noise across \(J\) vs \(J^c\), relate masked prediction risk to denoising the latent signal (cite standard blind-spot references; see `reference.md`).
6. **Reality gap** — Note when independence or stationarity fails for DWI (shared reconstruction artifacts, eddy-current structure, etc.).

## Alignment with this repository

- **Discussion and review prose:** [`src/discussion/README.md`](../../../src/discussion/README.md) — index and conventions.
- **Canonical technical report (Scheme 1 / Scheme 2):** [`src/J_invariance_DWMRI_denoising_report.md`](../../../src/J_invariance_DWMRI_denoising_report.md).
- **Scheme 2 (hybrid) implementation patterns:** e.g. [`src/restormer_hybrid/data.py`](../../../src/restormer_hybrid/data.py) (masked target channel in input stack), [`src/restormer_hybrid/fit.py`](../../../src/restormer_hybrid/fit.py) (masked MSE on occluded sites). Similar patterns exist under `drcnet_hybrid`, `drcnet_hybrid_multiple_networks`, `drcnet_hybrid_tl`, `unet3d_hybrid`.

When verifying behavior, **read the cited files**; do not rely only on this skill for line-level accuracy.

## Low-data playbook

Separate **theory** vs **implemented in repo**:

| Idea | Role |
|------|------|
| Self-supervision (J-invariant / blind-spot) | Uses single noisy dataset; no clean target required. |
| Masking probability / schedule | More occlusion → stronger invariance signal, higher bias if too aggressive; tune `mask_p` in hybrid configs. |
| Transfer between volumes or shells | Reduces per-scenario data need; see `drcnet_hybrid_tl` and related transfer-learning docs. |
| One model per target volume | `drcnet_hybrid_multiple_networks` — specialized but more total compute. |
| Smaller models / strong 3D inductive bias | Fewer samples to reach stable training; match capacity to patch size and gradient count. |
| Ensembling (masks, dropout) | Variance reduction at inference (Self2Self-style); optional add-on. |
| Physics-aware augmentation | Only where it preserves the assumed signal–noise structure; avoid breaking angular relationships if the model exploits them. |

If recommending supervised pretraining on external clean MRI, label it as **not** J-invariant self-supervision unless the user accepts that scope change.

## Citations

Use [`reference.md`](reference.md) for canonical papers and short “when to cite” notes.
