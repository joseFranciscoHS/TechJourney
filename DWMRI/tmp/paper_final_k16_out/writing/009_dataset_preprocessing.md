# 009 — Dataset and Preprocessing Details

## Context for the LLM

You are helping complete the **Materials and Methods** section of a research paper on self-supervised DW-MRI denoising. The current draft has a TODO at line 76 requesting detailed dataset and preprocessing specifications.

## Files to attach

Attach these files when you send this prompt. Paths are relative to `DWMRI/`.

| File | Purpose |
|------|---------|
| `paper/Sepulveda_dwmri_restormer.tex` | Current manuscript with TODO at line 76 |

## Prompt

Write the detailed dataset and preprocessing specification to replace the TODO at line 76 of the manuscript. Use the information below.

### D-Brain Dataset

**Citation**: Perrone D, Jeurissen B, Aelterman J, Roine T, Sijbers J, Pizurica A, et al. (2016) D-BRAIN: Anatomically Accurate Simulated Diffusion MRI Brain Data. PLoS ONE 11(3): e0149778. https://doi.org/10.1371/journal.pone.0149778

**Specifications**:
- **Type**: Synthetic digital phantom derived from anatomically accurate brain structures
- **Acquisition simulation**: 3T scanner simulation
- **Voxel size**: 1.4 × 1.4 × 1.4 mm³
- **Volumes**: 6 b=0 s/mm² volumes + 60 diffusion-weighted volumes at b=2500 s/mm²
- **Gradient directions**: 60 uniformly distributed on the unit sphere
- **Spatial dimensions**: 128 × 128 × 96 voxels (after cropping from full phantom)
- **Data availability**: Downloaded from Figshare (https://doi.org/10.6084/m9.figshare.2199001.v3)
- **Train/val/test split**: Not applicable (self-supervised training on single subject; evaluation against clean phantom ground truth)

### Stanford HARDI Dataset

**Citation**: Rokem A, Yeatman JD, Pestilli F, Kay KN, Mezer A, van der Walt S, Wandell BA (2015) Evaluating the Accuracy of Diffusion MRI Models in White Matter. PLoS ONE 10(4): e0123272. https://doi.org/10.1371/journal.pone.0123272

**Specifications**:
- **Type**: In vivo human brain acquisition
- **Scanner**: GE Discovery MR750 3T
- **Acquisition**: High angular resolution diffusion imaging (HARDI)
- **Volumes**: 10 b=0 s/mm² volumes + 150 diffusion-weighted volumes at b=2000 s/mm²
- **Gradient directions**: 150 uniformly distributed
- **Data availability**: DIPY built-in dataset (`dipy.data.fetch_stanford_hardi`)
- **Use case**: Real-scanner feasibility demonstration without ground truth

### Preprocessing Pipeline

1. **Normalization**: Per-volume min-max normalization to [0, 1] range independently for each diffusion-weighted volume
   - Formula: `normalized[..., i] = (volume[..., i] - min) / (max - min + ε)` where ε = 10⁻⁶

2. **Noise addition (D-Brain only)**: Synthetic Rician noise applied to normalized volumes (see Evaluation Protocol section for formula)

3. **Patch extraction**: 
   - D-Brain: 3D patches extracted from normalized volumes with progressive training schedule
   - Stage 1: 32³ patches, stride 8
   - Stage 2: 24³ patches, stride 6  
   - Stage 3: 16³ patches, stride 4

4. **Patch filtering**: Background patches excluded using threshold criterion
   - Patches where `max(clean_patch) ≤ 0.02` are discarded to avoid training on empty regions

5. **Brain mask (ROI) construction**: 
   - ROI defined as voxels where any diffusion volume exceeds intensity threshold 0.02
   - Used for computing ROI-specific metrics (PSNR-ROI, SSIM-ROI, FA-MAE, etc.)

### Implementation

Preprocessing code: `src/utils/data.py` (normalization functions), `src/drcnet_hybrid_rgs/data.py` (patch extraction and filtering)

Configuration: `src/drcnet_hybrid_rgs/config.yaml` and `src/restormer_hybrid_rgs/config.yaml`

## Expected output

A LaTeX paragraph (150-250 words) with inline citations that specifies:
- Subject count and data type for each dataset
- Scanner/acquisition parameters
- Voxel size and number of volumes
- Preprocessing steps with sufficient detail for reproducibility
- Brain mask definition
- Dataset citations in `\cite{}` format

The text should integrate smoothly into the existing manuscript at line 76, following the notation established in §2.1 (Data Representation).
