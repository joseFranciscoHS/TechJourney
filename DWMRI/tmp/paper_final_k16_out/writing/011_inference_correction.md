# 011 — Inference Method Correction

## Context for the LLM

You are correcting a **critical error** in the **Inference** subsection (§2.4) of a research paper on self-supervised DW-MRI denoising. The current manuscript describes a masked-voxel-only accumulation scheme, but the actual implementation averages full-volume predictions.

## Files to attach

| File | Purpose |
|------|---------|
| `paper/Sepulveda_dwmri_restormer.tex` | Current manuscript with INCORRECT equation at lines 115-119 |
| `src/drcnet_hybrid_rgs/reconstruction.py` | Actual inference implementation (lines 156-181) |

## Prompt

**Replace the entire §2.4 Inference subsection** (lines 113-122) with a corrected description. The current text claims:

> "To preserve the blind-spot interpretation at reconstruction time, the most conservative estimator averages only predictions for voxel locations that are masked in the corresponding stochastic pass"

This is **wrong**. The code shows:

```python
acc = acc + out.sum(dim=0)  # accumulates entire output volume
sum_preds[vol_k] = (acc / denom)  # denom = n_context * n_preds
```

### Corrected subsection text

**§2.4 Inference**

At inference time, each target volume is reconstructed through a stochastic ensemble procedure. For each target gradient $t$, the method samples $N_c$ random context sets $C_t^{(1)}, \ldots, C_t^{(N_c)}$ from the non-target gradients. For each context, $N_p$ spatial masks are drawn independently, forming masked target inputs and producing $N_p$ full-volume predictions. The final reconstruction averages all predictions:

$$
\bar{y}_t = \frac{1}{N_c \cdot N_p} \sum_{a=1}^{N_c} \sum_{b=1}^{N_p} f_\theta\left( z_t^{(a,b)} \right),
$$

where $z_t^{(a,b)}$ denotes the input formed from context $C_t^{(a)}$ and spatial mask $m^{(a,b)}$, and $f_\theta$ is the trained denoiser. This procedure is equivalent to Monte Carlo dropout applied to both the angular sampling dimension (via RGS) and the spatial masking dimension (via Bernoulli occlusion). Unlike the masked-voxel-only estimator that would preserve strict per-voxel J-invariance at test time, this implementation averages complete spatial predictions, reducing variance at the cost of including unmasked-site information in the ensemble. The inference cost scales as $O(G \cdot N_c \cdot N_p)$ forward passes, where $G$ is the number of diffusion gradients.

**Default hyperparameters**: D-Brain experiments use $N_c = 16$ context samples and $N_p = 12$ spatial mask draws per context, totaling 192 forward passes per target volume. Stanford HARDI uses $N_c = 16$ and $N_p = 23$ due to the larger gradient shell ($G=150$).

### TODO removal

Delete the TODO at line 122:

~~TODO: Confirm whether the current reconstruction code accumulates only masked-voxel predictions as written here or averages full-volume predictions from every stochastic pass.~~

This is now confirmed: the code averages full-volume predictions.

## Expected output

A replacement LaTeX subsection (200-250 words) that:
1. Describes the actual Monte Carlo ensemble averaging over contexts and masks
2. Provides the correct mathematical formula (simple average, not masked-site weighting)
3. Notes the departure from strict voxel-wise J-invariance at inference
4. States default $N_c$ and $N_p$ values
5. Removes the TODO

The tone should be factual and precise, avoiding defensiveness about the departure from the strictest J-invariant estimator.
