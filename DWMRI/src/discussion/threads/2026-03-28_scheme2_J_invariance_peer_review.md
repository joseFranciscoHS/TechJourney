# Thread: Scheme 2 J-invariance peer review

**Date:** 2026-03-28  
**Topic:** MD-S2S style (spatial–angular hybrid) — is it J-invariant?

## References

- Technical report: [`../../J_invariance_DWMRI_denoising_report.md`](../../J_invariance_DWMRI_denoising_report.md) (Section 4, Scheme 2)
- Implementation (example): [`../../restormer_hybrid/data.py`](../../restormer_hybrid/data.py), [`../../restormer_hybrid/fit.py`](../../restormer_hybrid/fit.py)
- Reconstruction (masked ensemble at inference): [`../../drcnet_hybrid/reconstruction.py`](../../drcnet_hybrid/reconstruction.py), [`../../restormer_hybrid/reconstruction.py`](../../restormer_hybrid/reconstruction.py)

## Question

Is the proposed Scheme 2 training objective J-invariant?

## Summary (peer-style answer)

**Verdict:** Yes, in the same sense as blind-spot / Noise2Void-style self-supervision, *given the constructed input*.

For each step, let the supervised set \(J\) be the **occluded** voxels of the target volume (mask zero there). Input uses \(\tilde{v}_j = M \odot v_j\) (zeros at \(J\)); other volumes \(v_{-j}\) are unmasked. Loss is masked MSE on occluded sites vs the **noisy** \(y_j\). Then \(\hat{v}_{j,J}\) does not depend on \(y_{j,J}\) through the input channel; supervision enters only via the loss.

This matches the implementation: masked target channel in the stack; loss on `(1 - mask)` in `fit.py`.

## Caveats / threats to validity

1. **Implementation leakage:** Normalization, pooling, or attention can mix information across voxels so predictions at \(J\) are no longer described by the ideal “only \(J^c\)” story—see **Elaboration** below.
2. **Terminology:** Section 1 calls \(J\) a “partition”; Scheme 2 uses a **random mask per sample**—clarify for reviewers.
3. **DWI noise / artifacts:** Independence assumptions are idealized (Rician, shared artifacts across gradients, motion). J-invariant *training* is still well defined; unbiased recovery of clean signal depends on how badly assumptions fail.
4. **Inference:** A **single** unmasked forward would shift the input distribution vs training; **this repo** instead uses **multiple masked forwards + average** (`n_preds`) for hybrid reconstruction—see **Elaboration: train–test gap** below.

### Elaboration: why “implementation leakage” matters

**What we want (ideal blind spot).** At occluded voxels \(J\), the value fed into the network is a **constant** (0) that does not change when \(y_{j,J}\) changes. Any function of that tensor alone then has outputs at \(J\) that are **invariant** to swapping \(y_{j,J}\) while holding \(v_{-j}\) and the **visible** part of \(v_j\) fixed.

**Where leakage enters.** Layers that compute each position’s activations using **aggregates over the whole patch (or batch)** can make those activations depend on *every* input value that entered the aggregate—including noisy values that were not zeroed because they sit on **visible** sites, or worse, statistics that still reflect the full tensor before masking in some architectures.

Concrete mechanisms:

- **Batch normalization (BN).** If mean/variance are taken over a batch of patches, statistics depend on all voxels in those patches, including visible noisy \(y_j\) and other channels. Within a single forward pass, **spatial** BN over \(H\times W\times D\) mixes all spatial locations: activations at an occluded site can depend on moments that include contributions from visible (noisy) voxels in the same channel. That does **not** inject \(y_{j,J}\) from the *masked* coordinates into the sum the same way raw input would, but it **does** couple predictions at \(J\) to the global spatial distribution of noise in the patch. Strict “conditional independence given only \(J^c\)” arguments in papers often assume pixelwise processing or masks constructed so no such mixing occurs; conv nets with spatial BN are a standard grey area.

- **Layer normalization / GroupNorm.** Often applied **per sample and per channel (and per group)** across **spatial** dimensions. Then moments are computed over all voxels in the patch for that channel—including both masked (0) and visible (noisy) sites. The normalized feature at an occluded location is therefore a **function of** the visible noisy values (through the mean/variance). That is the clearest form of **leakage**: \(\hat{v}_{j,J}\) can depend on \(y_{j,J^c}\) in a highly nonlinear way (expected and desired for denoising), but also on **moments** that entangle signal and noise across the patch. The classical Noise2Void argument is about independence of **noise at \(J\)** from predictors; entangling **noise on \(J^c\)** into norm stats can bias or correlate errors in subtle ways versus the idealized proof.

- **Global pooling + MLP / attention pooling.** Any step that produces a **single vector** summarizing the whole patch and broadcasts it back (or conditions attention on it) creates a path from every input voxel (including visible noisy \(y_j\)) to every output voxel, including \(J\). Again, this is not “reading \(y_{j,J}\) directly,” but it weakens the crisp statement “prediction at \(J\) uses only \(J^c\).”

- **Self-attention within the patch.** If queries at occluded locations can attend to **key/value** at visible locations, that is **intended** use of \(J^c\). If attention is **global** within the patch, keys at visible sites carry \(y_{j,J^c}\) (signal + noise). That is still blind-spot–compatible **if** keys never use \(y_{j,J}\). But if any bug or architectural path uses **unmasked** \(v_j\) in a branch, or if positional encodings plus softmax create effective sensitivity to masked positions’ *pre-mask* values (should not happen if zeroed), review the actual forward graph.

**Practical takeaway.** For a paper or grant review, it is honest to say: *J-invariance holds for the **stated input construction**; the network is an approximate blind-spot predictor. Normalization and global mixing may deviate from the strongest theoretical ideal; we rely on empirical behavior and optionally ablate (e.g. GroupNorm vs InstanceNorm per slice, or norm-free blocks) if reviewers push.*

### Elaboration: train–test gap at inference

**Two inference patterns (don’t confuse them).**

1. **Single forward, fully unmasked target channel.** Here each test forward uses the **full** noisy \(v_j\) everywhere (no zeros). That input layout was **not** seen during training on masked sites, so there is a real **distribution shift** relative to training forwards.

2. **This repository (hybrid DRCNet / Restormer).** `reconstruct_dwis` keeps the **training-style input**: for each target volume \(j\), it runs **`n_preds` forwards** with a **fresh Bernoulli mask** on \(v_j\) only (same `mask_p` idea as training), accumulates the predicted volume, then **averages** (`sum_preds / n_preds`). See e.g. [`drcnet_hybrid/reconstruction.py`](../../drcnet_hybrid/reconstruction.py) (loop over `n_preds`, mask target channel, forward, accumulate) and the sliding-window variant in [`restormer_hybrid/reconstruction.py`](../../restormer_hybrid/reconstruction.py). So the **optional** Self2Self-style “many masks + average” mitigation is **implemented here**, not only mentioned in theory: each forward stays in the **masked-target** regime; variance is reduced by Monte Carlo over masks.

**Residual gap even with ensembling.** Training still optimizes **masked MSE on occluded voxels only**; inference averages **full-volume** model outputs (one scalar field per forward). The **loss** definition at train time still differs from “every voxel is a supervised blind spot in one pass”—but the **input** distribution per forward is much closer to training than a single unmasked forward would be.

**Does J-invariance break?** Still no: J-invariance characterizes **training**. Inference is whatever protocol you choose; the hybrid reconstruction protocol above is deliberately aligned with training.

**Why reviewers still care.** (1) If a pipeline used **one** unmasked forward, point out shift in activations (norms, etc.). (2) Risk identities are for the **training** objective; ensemble inference is **variance reduction** + matched input stats, justified empirically. (3) Cost: `n_preds` forwards per volume (config e.g. `reconstruct.n_preds` in Restormer config).

**What to write in a paper.** *Training uses masked target channels and masked MSE on occluded voxels. At inference we run multiple stochastic masks on the target channel (matching training inputs), average the predicted volumes, and optionally use sliding-window blending for full volumes—reducing variance and keeping inputs closer to the training regime than a single fully unmasked forward.*

## Follow-ups (optional)

- [ ] Short paragraph in the technical report: random \(J\) vs fixed partition; inference vs training input.
- [ ] Audit `restormer_hybrid` (and other hybrid nets) for norm layers that mix statistics across masked/unmasked sites in a way that could leak \(y_{j,J}\).
