# Paper Banana prompt: DRCNet3D

**Output file:** `paper/figures/arch_drcnet3d.png`

## Prompt (paste below)

```text
Create a single-panel neural network architecture diagram titled "DRCNet3D Architecture".

Purpose: a 3D convolutional encoder–decoder that maps a multi-channel volumetric input to a single-channel volumetric output.

Layout: one vertical flowchart, top to bottom. Draw only this network — no side panels, no second model, no comparison cards.

Blocks and flow (top to bottom, in order):
1. Input box: "Input K ch"
2. Conv3d 3×3×3
3. Conv3d 2×2×2, stride 2 (downsample)
4. Yellow box: "FiLM (optional)" with small side label "[cos_x, cos_y, cos_z, b_norm]"
5. Orange box: "RCDB ×4 iterations". Inside the box also show: "GRU gates u_t, r_t" and "factorized 3×1×1, 1×3×1, 1×1×3 → fuse 1×1×1"
6. ConvTranspose3d 2×2×2, stride 2 (upsample)
7. Yellow box: "FiLM (optional)"
8. Box: "Concat with skip"
9. Conv3d 1×1×1
10. Conv3d 3×3×3
11. "PReLU / Sigmoid"
12. Output box: "Output 1 ch"

Skip connection: dashed gray arrow from the first Conv3d 3×3×3 to the "Concat with skip" box.

Footer notes (small text under the output):
- "Feature width ≈ 32"
- "No residual from input to output"

Colors: main blocks light blue; RCDB light orange; FiLM yellow; skip dashed gray; white background; flat fills; thin black outlines; clear sans-serif labels.

Style: clean academic vector schematic, publication-ready. No cartoons, no glossy 3D, no drop shadows, no brain or MRI photos, no parameter-count badges.
```

## Checklist after generation

- [ ] Vertical single-level encoder–decoder (one down, one up)
- [ ] Input K ch → Output 1 ch
- [ ] RCDB ×4 with GRU gates and factorized kernels
- [ ] Two yellow FiLM boxes; skip to concat; no input→output residual
