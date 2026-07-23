# Paper Banana prompt: Restormer-2D

**Output file:** `paper/figures/arch_restormer2d.png`

## Prompt (paste below)

```text
Create a single-panel neural network architecture diagram titled "Restormer-2D".

Purpose: a three-level 2D transformer encoder–decoder that processes multi-channel 2D images (slice-wise) and outputs a single-channel 2D image.

Layout: compact U-Net style — encoder on the left, bottleneck at the bottom, decoder on the right. Draw only this network. The title must be exactly "Restormer-2D".

Blocks and flow:
1. Input: "Input K ch"
2. "Overlap Patch Embed (3×3 conv)"
3. Encoder L1: "N=1, h=1, 12 ch, MDTA+GDFN (2D)"
4. "Down 2× (PixelUnshuffle)"
5. Encoder L2: "N=2, h=2, 24 ch" with yellow "FiLM (optional)"
6. "Down 2× (PixelUnshuffle)"
7. Latent (light purple): "N=2, 48 ch" with yellow "FiLM (optional)" and side label "[cos_x, cos_y, cos_z, b_norm]"
8. "Up 2× (PixelShuffle)"
9. Decoder L2: "N=2, h=2" with yellow "FiLM (optional)"
10. "Up 2× (PixelShuffle)"
11. Decoder L1: "N=1, h=1"
12. "Refine N=2"
13. Output head: "Conv + PReLU, γ·x + β"
14. Output: "Output 1 ch"

Skip connections: dashed gray arrows Enc L1 → Dec L1 and Enc L2 → Dec L2.

Also label channel widths: 12 → 24 → 48.
Footer notes:
- "Slice-wise 2D processing"
- "No residual from input to output"

Colors: encoder light blue; decoder light green; latent light purple; FiLM yellow; skips dashed gray; white background; flat fills; thin outlines; clear labels.

Style: clean academic vector schematic, publication-ready. No cartoons, no glossy 3D, no MRI photos, no parameter-count badges, no second network in the figure.
```

## Checklist after generation

- [ ] Title is Restormer-2D
- [ ] Widths 12/24/48; blocks (1,2,2); heads (1,2,2); refine N=2
- [ ] Labels say 2D MDTA+GDFN; Down/Up say PixelUnshuffle / PixelShuffle
- [ ] Slice-wise note; K→1; no input residual
