# Paper Banana prompt: Res-CNN-2D

**Output file:** `paper/figures/arch_res_cnn2d.png`

## Prompt (paste below)

```text
Create a single-panel neural network architecture diagram titled "Res-CNN-2D".

Purpose: a lightweight residual 2D CNN that maps a multi-channel 2D input to a single-channel 2D output.

Layout: short vertical flowchart. Draw only this network.

Blocks and flow:
1. Input: "Input K ch"
2. Embed: "Conv2d 3×3, K → dim (dim ≈ 24)"
3. Orange (or light blue) box: "Residual blocks ×2" with subtitle "Conv–GELU–Conv" and a visible curved residual (+) shortcut arrow around the block
4. Projection: "Conv2d 3×3, dim → 1"
5. Output: "Output 1 ch"

Footer notes:
- "No attention"
- "Slice-wise 2D processing"

Colors: embed and projection light blue; residual stack light orange preferred; white background; thin outlines; clear labels.

Style: clean academic vector schematic, publication-ready. No multi-level U-Net, no FiLM, no attention blocks, no cartoons, no MRI photos, no parameter-count badges.
```

## Checklist after generation

- [ ] Embed → residual Conv–GELU–Conv ×2 → 1-ch projection
- [ ] Residual shortcut visible
- [ ] Input K ch → Output 1 ch; no attention / U-Net / FiLM
