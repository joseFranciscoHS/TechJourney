# Paper Banana prompt: Restormer3D

**Output file:** `paper/figures/arch_restormer3d.png`

## Prompt (paste below)

```text
Create a single-panel neural network architecture diagram titled "Restormer3D".

Purpose: a three-level 3D transformer encoder–decoder that maps a multi-channel volumetric input to a single-channel volumetric output.

Layout: compact U-Net style — encoder column on the left, bottleneck at the bottom, decoder column on the right. Draw only this network.

Blocks and flow:
1. Input: "Input K ch"
2. "Overlap Patch Embed (3×3×3 conv)"
3. Encoder L1: "N=1, h=1, 12 ch, MDTA+GDFN (3D)"
4. "Down 2× (strided Conv3d)"
5. Encoder L2: "N=2, h=2, 24 ch" with a yellow "FiLM (optional)" box beside it
6. "Down 2× (strided Conv3d)"
7. Latent (light purple): "N=2, 48 ch" with yellow "FiLM (optional)" and side label "[cos_x, cos_y, cos_z, b_norm]"
8. "Up 2× (ConvTranspose3d)"
9. Decoder L2: "N=2, h=2" with yellow "FiLM (optional)"
10. "Up 2× (ConvTranspose3d)"
11. Decoder L1: "N=1, h=1"
12. "Refine N=2"
13. Output head: "Conv + PReLU, γ·x + β"
14. Output: "Output 1 ch"

Skip connections: dashed gray arrows Enc L1 → Dec L1 and Enc L2 → Dec L2.

Also label channel widths on the encoder path: 12 → 24 → 48.
Footer note: "No residual from input to output".

Colors: encoder light blue; decoder light green; latent light purple; FiLM yellow; skips dashed gray; white background; flat fills; thin outlines; clear labels.

Style: clean academic vector schematic, publication-ready. No cartoons, no glossy 3D, no MRI photos, no parameter-count badges, no second network in the figure.
```

## Checklist after generation

- [ ] Three levels; widths 12 / 24 / 48; blocks (1,2,2); heads (1,2,2); refine N=2
- [ ] MDTA+GDFN labeled 3D; Down/Up are strided / transposed 3D conv
- [ ] FiLM at Enc L2, latent, Dec L2; skips present; K→1; no input residual
