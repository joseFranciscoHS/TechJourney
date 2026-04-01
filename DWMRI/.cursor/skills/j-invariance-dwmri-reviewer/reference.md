# References for J-invariant / self-supervised denoising

Use these when peer-reviewing methods text or suggesting what to cite in a paper.

## Blind-spot / J-invariance (single modality)

- **Noise2Void** — Batson & Royer, *Noise2Void: Learning Denoising from Single Noisy Images* (2019). **When:** Voxel- or pixel-level masking; prediction at masked sites must not use noisy values at those sites; foundational for masked-input self-supervision.
- **Noise2Self** — Batson & Royer, *Noise2Self: Blind Denoising by Self-Supervision* (2019). **When:** Generalization beyond a single mask pattern; J-invariance as a unifying view.

## MRI / multi-channel and DWI-relevant

- **Patch2Self** — Fadnavis et al., *Patch2Self: Denoising diffusion MRI with self-supervised learning* (2020/2021). **When:** Angular redundancy; predicting one gradient volume from others; volume-level conditioning (related to Scheme 1 in this project’s report).

## Variance reduction (single image)

- **Self2Self** — Quan et al., *Self2Self With Dropout: Learning Self-Supervised Denoising From Single Image* (2020). **When:** Bernoulli masks + dropout ensemble at test time; variance–bias discussion for single-sample training.

## Project mapping (non-exhaustive)

| Topic | Suggested anchor |
|-------|------------------|
| Masked voxel in target volume + other volumes as input | Noise2Void + Patch2Self (hybrid / Scheme 2 narrative) |
| Predict volume \(j\) from \(v_{-j}\) only | Patch2Self |
| Mask ensemble / dropout | Self2Self |

Always match citation to the **exact training objective** implemented (masked pixels, which channels enter the network, loss mask convention).
