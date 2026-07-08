# 022 — Restormer Architecture Modifications for Hybrid RGS

## Context for the LLM

You are helping revise a research paper on **self-supervised denoising of Diffusion-Weighted MRI (DWI)** using a method called **Hybrid RGS** (Random Gradient Subset). The paper evaluates two neural backbone architectures: **DRCNet** (gated 3D CNN) and **Restormer-hybrid-RGS** (a 3D transformer adaptation).

The current manuscript [`paper/Sepulveda_dwmri_restormer.tex`](../../paper/Sepulveda_dwmri_restormer.tex) describes Restormer-hybrid-RGS in a single generic paragraph (`\subsubsection{Restormer-hybrid-RGS}`, lines 108-109) with no architectural detail. Additionally, Section 4.6 (3D vs 2D Convolutions ablation, lines 259-268) vaguely states "Restormer is evaluated with two-dimensional per-slice feature processing" — which is misleading, since the actual 2D "Restormer" implementation is a lightweight residual CNN, not a transformer.

This prompt asks you to:

1. **Expand the Restormer-hybrid-RGS subsubsection** with a detailed, structured comparison against the original 2D Restormer architecture from Zamir et al. (CVPR 2022).
2. **Add a new subsubsection** under Section 3.4 (Architecture-Agnostic Instantiations) describing the 2D variants used in the 3D-vs-2D ablation.
3. **Patch the vague sentence** in Section 4.6 to accurately reference the 2D variants and connect to the parameter counts in `tab:3d_vs_2d`.

**Key architectural differences to explain (from code analysis):**

### Restormer3D (3D port for Hybrid RGS) vs Original Restormer (2D)

- **Dimensionality**: All 2D operations (`Conv2d`, `rearrange("b c h w -> ...")`) converted to 3D equivalents (`Conv3d`, `rearrange("b c d h w -> ...")`). MDTA attention and GDFN feed-forward blocks structurally identical, just volumetric.
  
- **Depth reduction**: Original uses 4-level hierarchy with `num_blocks=[4,6,6,8]`, `heads=[1,2,4,8]`, `dim=48`, and 3 downsampling operations (bottleneck compression 512x). Hybrid RGS uses 3-level hierarchy with `num_blocks=[1,2,2]`, `heads=[1,2,2]`, `dim=12`, `ffn_expansion_factor=1.5`, and 2 downsampling operations (bottleneck compression 64x). This is a capacity/memory tradeoff for volumetric transformer attention on 3D DWI patches.

- **Resampling operators**: Original uses `Conv2d` + `PixelUnshuffle(2)`/`PixelShuffle(2)` (parameter-light, lossless channel-space reshuffle). The 3D port uses strided `Conv3d` (`Downsample3D`) and `ConvTranspose3d` (`Upsample3D`) because PyTorch lacks native 3D pixel-(un)shuffle primitives.

- **Input/output stem**: Original has `inp_channels=out_channels=3` (RGB restoration, or 6 for dual-pixel defocus deblurring). Hybrid RGS uses `inp_channels=K` (context gradients + Bernoulli-masked target channel at fixed index `K-1`) and `out_channels=1` (single denoised target volume).

- **Global input residual**: Original adds `+ inp_img` at the output (channel counts match: 3→3). This is impossible in Hybrid RGS (`K` in vs `1` out), so the model instead applies a learned global affine transformation (`output_scale`, `output_shift`) plus a `PReLU` or `Sigmoid` output activation (absent in the original).

- **FiLM conditioning** (new, not in original Restormer): Optional gradient-direction conditioning via `utils/film_layer.py` `FiLMLayer` modules inserted at 3 stages (after encoder level 2, at the latent bottleneck, after decoder level 2). Conditioned on the target channel's 4D acquisition vector `[cos_x, cos_y, cos_z, b_norm]` with identity-initialized MLP (γ≈1, β≈0). Evaluated in Table `\ref{tab:film_ablation}` as an exploratory extension.

### 2D Ablation Variants (Section 4.6)

**Important**: The `Restormer2D` implementation in `src/restormer_hybrid_rgs/model2d.py` is **not** a slice-wise transformer port. It is a small residual CNN:

- Architecture: `Conv2d` embed → N residual blocks of `[Conv2d-GELU-Conv2d]` + skip connection → `Conv2d` output
- No attention mechanism, no U-Net hierarchy, no MDTA/GDFN blocks
- This explains the ~12x parameter gap: 0.015M (2D) vs 0.178M (3D) already reported in `tab:3d_vs_2d`

Similarly, `DRCNet2D` is a simplified 2D convolutional variant of the 3D gated DRCNet backbone.

The paper's current description ("Restormer is evaluated with two-dimensional per-slice feature processing") is misleading and must be corrected to state that the 2D variants are lightweight convolutional baselines, not slice-wise transformer ports.

---

## Files to attach

Attach these files when you send this prompt. Paths are relative to `DWMRI/`.

| File | Purpose |
|------|---------|
| `paper/Sepulveda_dwmri_restormer.tex` | Current manuscript to patch |
| `src/restormer_hybrid_rgs/model.py` | `Restormer3D` implementation (3D transformer for Hybrid RGS) |
| `src/restormer_hybrid_rgs/model2d.py` | `Restormer2D` implementation (lightweight 2D residual CNN) |
| `src/drcnet_hybrid_rgs/model2d.py` | `DRCNet2D` implementation (for symmetry/comparison) |
| `src/utils/film_layer.py` | FiLM conditioning layer implementation |
| `src/restormer_hybrid_rgs/config.yaml` | Hyperparameters (`dim=12`, `num_blocks=[1,2,2]`, `heads=[1,2,2]`, etc.) |

---

## Reference: Original Restormer (2D)

For context, here is the relevant code from the original 2D Restormer (Zamir et al., CVPR 2022):

```python
# From https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py

class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        dual_pixel_task = False
    ):
        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(...) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)  # Conv2d + PixelUnshuffle(2)
        
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(...) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim*2**1))
        
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(...) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim*2**2))
        
        self.latent = nn.Sequential(*[TransformerBlock(...) for i in range(num_blocks[3])])
        
        # Decoder with Upsample (Conv2d + PixelShuffle(2)) and skip connections
        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(...)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(...)])
        
        # ... similar for level 2 and level 1
        
        self.refinement = nn.Sequential(*[TransformerBlock(...) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # ... encoder-decoder forward pass
        out_dec_level1 = self.output(out_dec_level1) + inp_img  # Global residual
        return out_dec_level1
```

Key differences in the 3D port: 4 levels → 3 levels, `Conv2d` → `Conv3d`, `PixelUnshuffle`/`PixelShuffle` → strided/transposed conv, `dim=48` → `dim=12`, global residual removed (channel mismatch), optional FiLM conditioning added.

---

## Prompt

Produce **three LaTeX snippets** to patch [`paper/Sepulveda_dwmri_restormer.tex`](../../paper/Sepulveda_dwmri_restormer.tex):

### 1. Rewrite `\subsubsection{Restormer-hybrid-RGS}` (currently lines 108-109)

Expand this into a detailed architectural description with the following structure:

**Paragraph 1 (motivation, ~80 words):** Restormer-hybrid-RGS adapts the original 2D Restormer architecture (Zamir et al., CVPR 2022) for 3D volumetric DWI denoising under the Hybrid RGS framework. The adaptation balances representational capacity with memory constraints for 3D transformer attention, while modifying the input/output interface to accommodate the `K`-channel random gradient subset sampling and blind-spot target masking.

**Paragraph 2 (itemized changes, ~270-350 words):** Describe the architectural modifications in a structured list or series of sentences covering:

- **Dimensionality extension**: All 2D spatial operations converted to 3D (convolutions, layer normalization reshape, attention spatial flattening). MDTA and GDFN blocks structurally unchanged, just volumetric.
  
- **Hierarchy depth reduction**: 4-level (3 downsamples, bottleneck 512x compression) → 3-level (2 downsamples, 64x compression). Specific config: `dim=12`, `num_blocks=[1,2,2]`, `heads=[1,2,2]`, `ffn_expansion_factor=1.5`. Cite the computational-memory tradeoff for 3D attention.

- **Downsampling/upsampling operators**: Original's `PixelUnshuffle`/`PixelShuffle` (lossless channel-space rearrangement) replaced with strided `Conv3d` and `ConvTranspose3d` due to lack of native 3D pixel-shuffle in PyTorch.

- **Input/output stem**: `inp_channels=K` (context + masked target), `out_channels=1` (single denoised volume), replacing the original's 3→3 RGB restoration interface.

- **Output head**: Global input residual (`+ inp_img`) removed due to channel mismatch; replaced with learned affine transformation (`output_scale`, `output_shift`) and `PReLU` or `Sigmoid` activation for bounded output.

- **Optional FiLM conditioning** (see Table `\ref{tab:film_ablation}`): Gradient-direction metadata (target channel's `[cos_x, cos_y, cos_z, b_norm]`) modulates features at three intermediate stages (encoder-2, bottleneck, decoder-2) via identity-initialized MLPs. Treated as an exploratory extension rather than core architecture.

**Optional comparison table (if space permits):** Small `tabular` with columns: Component | Original Restormer (2D) | Restormer3D-hybrid-RGS. Rows: Hierarchy depth, Spatial dim, Resampling, I/O channels, Output residual.

**Tone**: Technical, specific, honest about tradeoffs (reduced depth = lower capacity, transposed conv = learnable but less parameter-efficient). Cross-reference Table `\ref{tab:3d_vs_2d}` for parameter counts (0.178M).

---

### 2. Add new `\subsubsection{2D Variants for Backbone-Dimensionality Ablation}` under Section 3.4

Insert this **after** the Restormer-hybrid-RGS subsubsection and **before** Section 3.5 (Inference). Target length: ~150-200 words.

**Content:**

The 3D-vs-2D ablation (Section 4.6, Table `\ref{tab:3d_vs_2d}`) evaluates whether volumetric processing is necessary for the Hybrid RGS objective or whether lightweight 2D slice-wise backbones suffice. To isolate architectural dimensionality from the self-supervised training framework, we implement 2D variants of both DRCNet and Restormer.

**DRCNet-2D** adapts the gated convolutional blocks to 2D operations, processing each axial slice independently with the same `K`-channel input structure (context gradients + masked target).

**Restormer-2D** is **not** a slice-wise transformer port. Instead, it is a lightweight residual CNN: `Conv2d` embedding → N residual blocks of `[Conv2d-GELU-Conv2d]` with skip connections → `Conv2d` output projection. This design omits attention mechanisms, hierarchical downsampling/upsampling, and MDTA/GDFN blocks entirely. The resulting model has approximately 0.015M parameters compared to 0.178M for Restormer3D (Table `\ref{tab:3d_vs_2d}`), making it a computational baseline for evaluating whether the 3D transformer's higher capacity justifies the 12x parameter cost.

Both 2D variants use the same Hybrid RGS training objective (random gradient sampling, Bernoulli target masking, masked loss) as their 3D counterparts, ensuring that any performance difference reflects architectural dimensionality rather than differences in self-supervision.

---

### 3. Patch Section 4.6 (3D vs 2D Convolutions) line ~262

**Find this sentence (approximately line 262):**

> Restormer is evaluated with two-dimensional per-slice feature processing.

**Replace with:**

> Restormer is evaluated with a lightweight 2D residual CNN variant (see Section 3.4) rather than a slice-wise transformer port.

Also find any other vague references to "2D processing" in this section and ensure they cite the new subsubsection or clarify that the 2D models are simplified baselines, not full transformer ports.

---

## Formatting requirements

- LaTeX source only (no `\begin{document}` or preamble).
- Use `\subsubsection{...}` for subsection headings.
- Cross-reference tables and sections with `\ref{...}`.
- Use `booktabs` style if you include the optional comparison table (`\toprule`, `\midrule`, `\bottomrule`).
- Cite the original Restormer paper as `\cite{restormer}` (already in bibliography).
- Keep tone consistent with the rest of the manuscript: cautious, precise, no overselling.
- Do **not** change any numeric results, parameter counts, or table contents — those are authoritative from the registry.

---

## Expected output

Return:

1. **Snippet 1**: Full replacement text for `\subsubsection{Restormer-hybrid-RGS}` (~350-450 words + optional table).
2. **Snippet 2**: Full text for new `\subsubsection{2D Variants for Backbone-Dimensionality Ablation}` (~150-200 words).
3. **Snippet 3**: Exact search-and-replace instruction for the sentence in Section 4.6.
4. **Location guide**: Brief list (3-5 bullets) of where each snippet goes in the manuscript (section number, approximate line, before/after which existing text).

Do not include the full paper — just the three replacement snippets.
