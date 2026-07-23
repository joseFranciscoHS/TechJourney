# Paper Banana prompt: Plain-CNN-2D

**Output file:** `paper/figures/arch_plain_cnn2d.png`

## Prompt (paste below)

```text
Create a single-panel neural network architecture diagram titled "Plain-CNN-2D".

Purpose: a minimal feed-forward 2D CNN that maps a multi-channel 2D input to a single-channel 2D output.

Layout: one short vertical stack only. Sparse and readable. Draw only this network.

Blocks and flow (exactly these stages):
1. Input: "Input K ch"
2. "Conv2d 3×3, K → 32, PReLU"
3. "Conv2d 3×3, 32 → 32, PReLU"
4. "Conv2d 3×3, 32 → 1"
5. Output: "Output 1 ch"

Footer notes (small text below the stack):
- "No skip connections"
- "No gating or recurrence"
- "No attention"
- "Slice-wise 2D processing"

Colors: all blocks light blue; white background; thin black outlines; clear sans-serif labels.

Style: clean academic vector schematic, publication-ready. Emphasize simplicity. No U-Net branching, no FiLM, no residual arrows, no cartoons, no MRI photos, no parameter-count badges.
```

## Checklist after generation

- [ ] Exactly three conv stages: K→32→32→1 with PReLU after the first two
- [ ] Linear stack only (no skips / FiLM / residuals / attention)
- [ ] Input K ch → Output 1 ch
