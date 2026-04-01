# Thread: Train on gradients 1–10, denoise 11–20 — contrast / “white” appearance

**Date:** 2026-03-28  
**Context:** Full DWMRI e.g. `(x, y, z, V)` with many volumes (example shape `(128,128,96,66)`). Goal: **less training data** — train a hybrid denoiser (e.g. 10 inputs → 1 output) only on **volumes 1–10**, then apply the **same checkpoint** to denoise **volumes 11–20** (same acquisition). Noise is visibly reduced on 11–20, but outputs share a similar **“white” contrast** to volumes 1–10.

## References

- Hybrid Scheme 2 report: [`../../J_invariance_DWMRI_denoising_report.md`](../../J_invariance_DWMRI_denoising_report.md)
- Reconstruction (MC masks): [`../../drcnet_hybrid/reconstruction.py`](../../drcnet_hybrid/reconstruction.py), [`../../restormer_hybrid/reconstruction.py`](../../restormer_hybrid/reconstruction.py)
- Output calibration note: [`../../restormer_hybrid/README.md`](../../restormer_hybrid/README.md) (`scale_and_shift`)

## Question

Is the “same white contrast as 1–10” mainly **overfitting**, or something else? Is cross-block transfer still a good idea?

## Summary (peer-style)

**Transfer can genuinely reduce noise** on held-out directions (same subject, shared anatomy, angular redundancy in the stack).

**Homogeneous “white” appearance on 11–20** is **plausible without invoking only noise memorization**:

1. **Conditional distribution shift:** The map was fit only when the **target** index lived in \(\{1,\ldots,10\}\) with context from that block. For target 15, the **angular layout** of the 10 input channels may never have been optimized; the network may still **suppress noise** while **mapping intensities** toward the **output regime** it learned on 1–10 (MSE / self-sup → shrinkage toward training marginals).

2. **Fixed output calibration:** Learned **`scale_and_shift`** (and optional **`rescale_to_01`**) are tuned on the **training** distribution; they **do not** adapt per new direction. That alone can **homogenize** apparent brightness across volumes at inference.

3. **Terminology:** “Overfitting to white contrast” is a reasonable **informal** label for **distribution shift + fixed affine readout**, not necessarily memorization of noise patterns.

## Clarify in write-ups / experiments

- How are the **10 context channels** chosen for a target in 11–20? (fixed 1–10, sliding window along the shell, etc.)
- **Normalization:** Same per-volume scaling for training vs transfer blocks?
- **Shell / \(b\)-value:** Are 1–10 and 11–20 strictly comparable?

## Ideas to try (not mutually exclusive)

- **Expose more directions during training** without using all at once: e.g. rotate which indices are targets / which 10 channels enter the stack (random subsets per epoch) so the head sees **full-shell contrast statistics**.
- **Light calibration** after denoising: per-volume affine fit (even on noisy references) to fix mean/variance; ablate vs raw net output.
- **Metrics:** Separate **noise reduction** from **intensity bias** (e.g. vs reference or vs directional marginals); FA / invariant summaries if the use case is tractography.
- **Baseline:** Few-epoch **fine-tune** on 11–20 — if contrast moves quickly, the issue is **transfer/calibration**, not capacity.

## Open status

- [ ] Document exact **context indexing** for targets outside the training block.
- [ ] Optional ablation: `scale_and_shift` off at inference vs on; per-volume rescaling modes in config.

---

## Closure (agreed design — random 10-tuples + matching inference)

**Training — what \(i_1,\ldots,i_{10}\) mean:** Index \(i_j\) is the **\(j\)-th random draw** from the pool of 60 volumes (e.g. sample 10 distinct indices without replacement, **order = draw order**, **not** sorted by acquisition index). The masked target sits at a **fixed channel** (e.g. always channel 10). Over many minibatches, **each physical volume** can appear in the target slot with balanced frequency (same idea as “randomly permute the 10 and fix mask at one channel”). This avoids the **sorted-last = always max index** trap.

**Inference — match training distribution:** For each target volume \(k\), use **several random 9-tuples of distinct indices from the remaining 59**, concatenate \(k\) at the **fixed target channel**, forward + **average** predictions (Monte Carlo over context sets), analogous in spirit to multiple random masks at inference. Cost scales with number of context samples; tune for speed vs variance.

**Sliding-window-only inference** is **not** aligned with this training scheme unless you also **mix sliding windows into training**; random-context inference is the consistent choice for the above training protocol.
