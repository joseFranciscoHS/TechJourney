Date: April 25, 2026

# Technical Report: J-Invariance and Hybrid RGS Self-Supervised Denoising for Diffusion-Weighted MRI (DWI)

This document synthesizes the J-invariance framework and its use in Self2Self (S2S) and Patch2Self (P2S). Section 4 specifies the **Hybrid RGS** (Random Gradient Subset) training and reconstruction scheme used in the implementations [`drcnet_hybrid_rgs`](drcnet_hybrid_rgs/) and [`restormer_hybrid_rgs`](restormer_hybrid_rgs/). For the earlier “two-scheme” discussion (volume-based P2S vs classic hybrid with all gradients), see [`J_invariance_DWMRI_denoising_report.md`](J_invariance_DWMRI_denoising_report.md).

---

## 1. The J-Invariance Theoretical Framework

J-invariance provides a general framework for denoising high-dimensional measurements without requiring clean training data (ground truth) or explicit noise models. A function $f$ is defined as *J-invariant* with respect to a partition $J$ of the data dimensions if the prediction along the dimensions in $J$ does not depend on the noisy input values in those same dimensions.

Under the assumption that noise is statistically independent across dimensions while the true signal exhibits correlation, minimizing the error between the prediction and the noisy data is equivalent to minimizing the error with respect to the clean signal. This allows the calibration of algorithms—from classical filters to deep neural networks—using only the available noisy information.

### Interpretation of J

1. **What are the dimensions in this context?**  
   The literature defines a noisy measurement $X$ as a vector in an $m$-dimensional space $\mathbb{R}^m$. These dimensions represent the coordinates or basic units of information that constitute the data. Depending on the domain, a dimension may be:
   - A single pixel in an image
   - A molecule detected in genetic sequencing
   - A complete 3D volume within a 4D DWI dataset

2. **What is J?**  
   $J$ is a subset of these dimensions. If the data are indexed $1, \ldots, m$, then $J$ is a partition of those indices into groups. Each group is a J-set.

3. **J dimensions in the methods studied**  
   - **MD-S2S (pixels):** The J dimensions are pixels occluded by a Bernoulli mask. The network must predict those pixels $f(x)_J$ using only the visible pixels (the complement $J^c$). It is not allowed to "see" the original noisy values of the occluded pixels.
   - **Patch2Self (volumes):** The J dimensions correspond to a specific volume $v_j$. The partition $J$ divides the 4D dataset into individual volumes. Being J-invariant here means that to reconstruct volume $j$, the algorithm may only use information from the other volumes $v_{-j}$; the original volume $j$ (the input in those dimensions) is ignored by the regressor.

### Bias–Variance Tradeoff in the J-Set

The structure of the partition $J$ is critical for model performance. There is an inherent tradeoff:

- **Bias:** If the J-set is too large, the model loses crucial input information (the complement $J^c$) needed to reconstruct the signal, which can increase systematic error or bias.
- **Variance:** Conversely, if $J$ is too small, more data are available for prediction, but the quality of that prediction is estimated from fewer noisy samples, which affects training stability.

The optimal choice of $J$ should reflect the patterns of signal dependence and noise independence in the specific domain (spatial or angular).

---

## 2. Implementation in Self2Self (S2S)

Self2Self (S2S) applies J-invariance to enable training on a single noisy image.

- **Statistical interpretation:** S2S treats the denoising network as a Bayes estimator whose accuracy is measured by Mean Squared Error. The main challenge in single-sample learning is a large increase in variance.
- **Invariance mechanism:** It uses Bernoulli masks to partition pixels, forcing the network to predict occluded pixels from their visible neighbors.
- **Variance reduction:** It implements a dropout-based ensemble. By averaging multiple predictions with different masks and dropout configurations at test time, the variance of the final prediction is reduced, improving the reconstructed image quality.

---

## 3. Implementation in Patch2Self (P2S)

Patch2Self adapts J-invariance to 4D Diffusion-Weighted MRI (DWI).

- **Invariance mechanism:** It defines the partition $J$ at the level of complete volumes (gradient directions). To denoise a volume $v_j$, the model uses only information from all other acquired volumes $v_{-j}$.
- **Angular redundancy:** It exploits the fact that, although gradient directions vary, the underlying anatomical structures are consistent across acquisitions.
- **Architecture:** It originally uses local linear regressors over 3D patches, assuming noise in each volume is statistically independent.

---

## 4. Proposed Training Scheme: Hybrid RGS

**Hybrid** here means the same statistical construction as the spatial–angular **Scheme 2** hybrid (all or a subset of gradient channels as input, Bernoulli masking on the **target** channel only, masked MSE on occluded voxels). **RGS** (Random Gradient Subset) means that, instead of always stacking all $G$ diffusion-weighted volumes in channel order, each training sample stacks **$K$** volumes sampled **without replacement** from the shell of size $G$, with a **fixed** channel index `target_channel` (typically $K-1$) for masking and loss. The implementations share the same dataset and loss logic in `data.py` / `fit.py`; they differ mainly in network architecture and in how full volumes are reconstructed from patches (Restormer uses tiled inference with blending).

### 4.1 Formal definition (notation aligned with code)

Let the acquired DWI shell (after discarding $N_{\mathrm{b0}}$ b0 volumes from the leading axis of the 4D array) consist of $G$ gradient volumes $\{y^{(g)}\}_{g=1}^G$, each $y^{(g)} \in \mathbb{R}^{X \times Y \times Z}$ (noisy). Hyperparameters:

| Symbol / config key | Role |
|---------------------|------|
| $G$ = `shell_gradient_volumes` | Number of DWI volumes in the cropped shell used for training/inference. |
| $K$ = `num_input_volumes` = `model.in_channel` | Number of channels stacked as network input; must match the model. |
| `target_channel` $\in \{0,\ldots,K-1\}$ | Channel where spatial masking is applied and where the scalar network output is supervised (implementation default: $K-1$). |
| `mask_p` | For each spatial location, `mask` is 1 with probability $1-\texttt{mask\_p}$ and 0 with probability `mask_p` (occluded sites). |

**One training sample (mode `shell_sampling_mode: rgs`):**

1. Choose a 3D patch window at coordinates $(x,y,z)$ with spatial size from `patch_size` / progressive stages (`step` controls the sliding-window grid).
2. Draw an ordered $K$-tuple of **distinct** indices $(g_1,\ldots,g_K)$ uniformly among all $K$-subsets of $\{1,\ldots,G\}$ without replacement (`numpy.random.default_rng().choice(G, size=K, replace=False)`). Stack patches into $\tilde{Y} \in \mathbb{R}^{K \times X_p \times Y_p \times Z_p}$ where channel $c$ holds $y^{(g_c)}$ restricted to the patch.
3. Draw a spatial Bernoulli mask $M \in \{0,1\}^{X_p \times Y_p \times Z_p}$ (same rule as above). Build input $\tilde{X}$ identical to $\tilde{Y}$ except at channel `target_channel` $= t$:  
   $$\tilde{X}_{t} = M \odot \tilde{Y}_{t}$$  
   (occluded voxels are multiplied by zero in the input; other channels are unmasked).
4. Network map $f_\theta: \mathbb{R}^{K \times X_p \times Y_p \times Z_p} \to \mathbb{R}^{1 \times X_p \times Y_p \times Z_p}$ outputs $\hat{y}$ supervised to match **channel** $t$ of $\tilde{Y}$, i.e. the physical gradient indexed by the $t$-th entry of the drawn $K$-tuple (in code: `window[target_channel]`). The stack order follows the random draw order, not a fixed q-space ordering of the shell.

**J-invariant set (per sample):** For fixed $M$ and fixed context, let $J$ be the set of voxel coordinates in the target channel where $M=0$. The network does not receive the original noisy values at those sites in that channel (they are zeroed). Supervision applies only at those sites via the noisy target $\tilde{Y}_t$ in the loss (see §4.2).

**Relation to sequential hybrid:** If `shell_sampling_mode` is `sequential`, each sample uses **all** $V=G$ volumes in fixed order, rotates the masked volume index $j \in \{0,\ldots,V-1\}$ across dataset indices, and matches the classic “all channels in, one volume out” hybrid. RGS generalizes the angular part by randomizing **which** $K$ gradients appear and their **stack order**, while keeping the mask on a **fixed** channel index $t$.

### 4.2 Self-supervised training objective

The training loss is **masked MSE** on occluded voxels only (code uses `(1 - mask)` as the loss support):

$$\mathcal{L} = \frac{\sum_{x,y,z} \bigl(1 - M_{xyz}\bigr)\,\bigl(\hat{y}_{xyz} - \tilde{Y}_{t,xyz}\bigr)^2}{\sum_{x,y,z} \bigl(1 - M_{xyz}\bigr)}\,.$$

Here $\tilde{Y}_{t}$ is the **noisy** patch on the target channel before masking. No gradient is applied on voxels where $M=1$ for that channel.

Optimization uses Adam with optional learning-rate schedules (`config.yaml`: e.g. StepLR, cosine warm restarts, ReduceLROnPlateau). Mixed precision (`use_amp`) is supported when training on CUDA.

### 4.3 Data layout, patch construction, and experimental knobs

**Volume axis after loading:** Arrays are stored as $(X,Y,Z,V)$. The dataset transposes to $(V,X,Y,Z)$ for patch extraction.

**Cropping the shell (`run.py`):** Non–b0 DWIs are taken from global volume index `num_b0s` up to (exclusive) `num_b0s + shell_gradient_volumes` when RGS is enabled, so the effective shell size is $G$. Optional spatial crops `take_x`, `take_y`, `take_z` restrict the field of view.

**Patch grid:** Valid patch origins are computed on the transposed volume array with patch shape $(K, p, p, p)$ in RGS mode (channel dimension $K$). `step` defines the sliding-window stride. Optional **patch filtering** (`patch_filter_method`): `none`, `threshold` (exclude patches whose clean reference patch has `max <= min_signal_threshold`), or `otsu` (exclude patches outside a brain mask from `median_otsu`).

**Dataset length:** In RGS mode, `__len__` equals the number of valid spatial windows (one random gradient subset per index). In sequential mode, length is multiplied by $V$ because the target volume index cycles.

**Progressive learning (optional, `train.progressive`):** Multiple stages with increasing `patch_size`, `step`, per-stage `batch_size`, and `epochs`. Each stage rebuilds the dataset and optimizer; the best checkpoint from the final stage is copied to `checkpoint_dir/best_loss_checkpoint.pth` for reconstruction.

**Training subset (implementation detail):** Progressive training in `drcnet_hybrid_rgs/run.py` uses a random subset of patch indices per stage (`subset_fraction = 0.6`). `restormer_hybrid_rgs/run.py` uses the full patch set in progressive mode (`subset_fraction = 1`) and may subsample in non-progressive mode depending on configuration—check the active `run.py` branch when reproducing experiments.

### 4.4 Reconstruction and inference

Inference must match the **channel semantics** learned during training: $K$ inputs, mask on `target_channel`, single-channel output for the volume placed in that slot.

**Sequential hybrid (`reconstruct_dwis`):** For each shell index $k$, form the full $V$-channel stack, apply independent spatial Bernoulli masks only to volume $k$, forward the network, and accumulate predictions. Average over `n_preds` mask draws per volume. (DRCNet path operates on full volumes in memory; Restormer uses **sliding windows** with overlap and **Gaussian separable blend weights** to merge patch predictions, and can batch mask draws via `pred_chunk_size`.)

**RGS hybrid (`reconstruct_dwis_rgs`):** For each true gradient index $k \in \{0,\ldots,G-1\}$:

1. **Outer Monte Carlo (context):** Repeat `n_context_samples` times: sample a set of $K-1$ distinct indices from $\{0,\ldots,G-1\}\setminus\{k\}$, build an ordered stack of length $K$ by appending $k$ at position `target_channel` (implementation concatenates `ctx` then `vol_k` so the target volume occupies the last slot when `target_channel = K-1`).
2. **Inner Monte Carlo (spatial mask):** Repeat `n_preds` times: apply the same Bernoulli masking rule as training on `target_channel` only, forward $f_\theta$, accumulate the scalar output into the accumulator for volume $k$.
3. **Average:** Divide by `n_context_samples * n_preds` (and, for Restormer patch inference, by the spatial weight map times the same denominator so overlapping tiles remain unbiased).

This design estimates the denoised volume by expectations over **random angular contexts** and **random blind-spot masks**, reducing variance from either source alone.

**Post-processing (metrics pipeline in `run.py`):** Optional `rescale_to_01` (e.g. `per_volume` or `match_gt`), `clip_to_range` to $[0,1]`, optional background median subtraction, and ROI metrics using `metrics_roi_threshold`.

### 4.5 Assumptions, threats to validity, and computational cost

**Statistical assumptions (as in classic hybrid / blind-spot training):** Conditional independence of noise across the voxels used as targets and the information used for prediction; sufficient signal correlation along the unmasked support (other gradients in the $K$-stack, plus visible voxels of the target channel) to identify the clean signal. **Violations:** shared structured noise across DWIs, motion, Gibbs artifacts, or Rician coupling can break the ideal J-invariance story and introduce bias.

**RGS-specific caveat:** Each sample uses a **random permutation** of $K$ distinct shell indices as channel order; `target_channel` fixes **which slot** is masked and supervised (typically $K-1$), while the physical gradient occupying that slot varies with the draw. If $K \ll G$, many tuples never appear in one forward during training; inference averages over many random $(K-1)$-tuple context draws (`n_context_samples`) to marginalize over angular neighbors when estimating each volume $k$.

**Complexity:** RGS inference scales roughly as $O\bigl(G \cdot \texttt{n\_context\_samples} \cdot \texttt{n\_preds} \cdot \text{forward}\bigr)$ per full volume set, multiplied by the number of spatial patches when using tiled Restormer reconstruction. Tuning `n_context_samples`, `n_preds`, and patch tiling is a speed–quality tradeoff.

### 4.6 Implementation variants (DRCNet vs Restormer)

| Aspect | `drcnet_hybrid_rgs` | `restormer_hybrid_rgs` |
|--------|---------------------|-------------------------|
| Backbone | `DenoiserNet` (gated / factorized 3D CNN) | `Restormer3D` (transformer-style 3D blocks) |
| Full-volume inference | Direct forward on cropped $(V,X,Y,Z)$ tensors | Tiled patches: `patch_size`, `overlap`, Gaussian blend weights |
| Memory controls | Whole-volume RGS loops | `pred_chunk_size` batches mask passes inside each patch tile |
| Config entry points | `drcnet_hybrid_rgs/config.yaml`, `python -m drcnet_hybrid_rgs.run` | `restormer_hybrid_rgs/config.yaml`, `python -m restormer_hybrid_rgs.run` |

The **statistical** training object (RGS subset, blind-spot mask, masked MSE) is the same; only the inductive bias and inference tiling differ.

---

## References in repository

- Package overview (modes, keys): [`drcnet_hybrid_rgs/README.md`](drcnet_hybrid_rgs/README.md)
- Dataset / mask / RGS sampling: [`drcnet_hybrid_rgs/data.py`](drcnet_hybrid_rgs/data.py), [`restormer_hybrid_rgs/data.py`](restormer_hybrid_rgs/data.py)
- Loss loop: [`drcnet_hybrid_rgs/fit.py`](drcnet_hybrid_rgs/fit.py), [`restormer_hybrid_rgs/fit.py`](restormer_hybrid_rgs/fit.py)
- Data crop + progressive + reconstruct dispatch: [`drcnet_hybrid_rgs/run.py`](drcnet_hybrid_rgs/run.py), [`restormer_hybrid_rgs/run.py`](restormer_hybrid_rgs/run.py)
- Reconstruction: [`drcnet_hybrid_rgs/reconstruction.py`](drcnet_hybrid_rgs/reconstruction.py), [`restormer_hybrid_rgs/reconstruction.py`](restormer_hybrid_rgs/reconstruction.py)
