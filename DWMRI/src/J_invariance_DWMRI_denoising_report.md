Date: December 22, 2025

# Technical Report: Foundations of J-Invariance and Denoising Proposals for Diffusion-Weighted MRI (DWI)

This document presents a synthesis of the J-invariance theoretical framework and its application to self-supervised denoising methods (Self2Self and Patch2Self). It also outlines the methodological proposals for the experimental phase of the research.

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

## 4. Proposed Training Schemes

To advance the research, two strategies are proposed using neural networks (NNs) with 3D convolutions to exploit the volumetric structure of DWI:

### Scheme 1: Generalized Patch2Self (Volume-Based)

- **Description:** A neural network architecture is used to learn the relationship between volumes, generalizing the original linear regression of Patch2Self.
- **Input:** Set of volumes $v_{-j}$ (e.g., 9 input volumes to predict volume 10).
- **Rationale:** By construction, the volume-based J-invariance suppresses noise independent to each acquisition while preserving the coherent structural signal across angular dimensions.

### Scheme 2: MD-S2S Style (Spatial–Angular Hybrid) — *active implementation*

This is the scheme implemented in the hybrid training pipelines (e.g. `drcnet_hybrid`, `restormer_hybrid`, `drcnet_hybrid_multiple_networks`): a single network takes **all** gradient volumes as input channels and predicts **one** target volume, combining Patch2Self-style angular conditioning with MD-S2S-style spatial masking on the target channel only.

#### Construction (per training sample)

Fix a 3D patch (sliding window) and a target volume index $j$. Let $M \in \{0,1\}^{X \times Y \times Z}$ be an independent Bernoulli draw over voxels (parameter set by `mask_p`: larger `mask_p` yields more occluded voxels).

- **Input to the network:** Stack $\tilde{v}_1,\ldots,\tilde{v}_N$ where $\tilde{v}_i = v_i$ for $i \neq j$, and $\tilde{v}_j = M \odot v_j$ (elementwise product). Occluded voxels in $v_j$ are set to zero in the input; all other volumes are unmasked.
- **Forward map:** $f_\theta: \mathbb{R}^{N \times X \times Y \times Z} \to \mathbb{R}^{1 \times X \times Y \times Z}$ produces a prediction $\hat{v}_j = f_\theta(\tilde{v})$ of the target volume.
- **Training loss (masked MSE):** With $(1-M)$ selecting the **occluded** voxels (supervision mask in code: loss on `1 - mask`),  
  $$\mathcal{L} = \frac{\sum_{x,y,z} (1-M_{xyz})\,\bigl(\hat{v}_{j,xyz} - y_{j,xyz}\bigr)^2}{\sum_{x,y,z} (1-M_{xyz})}\,,$$  
  where $y_j$ is the **noisy** observed target volume on that patch (the same $v_j$ before masking). No loss is applied on visible voxels of $v_j$ in this formulation.

#### Why this is J-invariant (for reviewers)

For a **fixed** mask $M$, let $J$ be the set of voxel indices $(x,y,z)$ in the target volume where $M_{xyz}=0$ (occluded in the input). The input values in those coordinates are identically zero and do not carry the original noisy measurements $y_{j,xyz}$. Under the usual idealization that the network’s prediction at $J$ is determined only by its input tensor, $\hat{v}_{j,J}$ does **not** depend on $y_{j,J}$—only on $v_{-j}$ and on $y_j$ at voxels where $M=1$. Supervision uses $y_{j,J}$ only in the loss, not as input. This matches the J-invariance pattern used in Noise2Void / blind-spot denoising and in Patch2Self, extended here to **two** sources of redundancy: across gradients ($v_{-j}$) and within-plane neighbors in $v_j$.

**Assumptions (explicit):** Unbiased, spatially and angularly independent noise (conditional on signal) so that the risk identity linking masked loss to denoising the underlying signal holds; signal correlation across $v_{-j}$ and across neighboring voxels in $v_j$ so that prediction from the complement is feasible. Violations (e.g. structured noise, Gibbs ringing shared across volumes) weaken the interpretation and can introduce bias.

#### Inference

At test time the implementation uses the **full** unmasked stack (or the project’s chosen reconstruction protocol); training masks are not applied. Optional variance reduction via multiple random masks or dropout ensembles follows the Self2Self spirit but is a separate design choice from J-invariance itself.

#### Relation to Scheme 1

Scheme 1 drops $\tilde{v}_j$ from the input entirely (pure volume-based J). Scheme 2 **adds** masked $\tilde{v}_j$ so the network can use same-volume spatial context while still supervising only at occluded voxels, which can reduce bias when angular redundancy alone is insufficient for a patch.
