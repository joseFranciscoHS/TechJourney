# Shared notes for authors (Paper Banana does not read this file)

Paste **only** the fenced prompt body from each `arch_*.md` into
[Scientific Diagram Maker](https://paperbanana.me/docs/scientific-diagram-maker).
Each paste block is self-contained; do not paste this file.

## Keep consistent across the five figures

| Role | Color |
|------|--------|
| Encoder / main blocks | Light blue |
| Decoder blocks | Light green |
| Bottleneck / latent | Light purple (Restormer only) |
| Recurrent / residual process | Light orange |
| FiLM | Yellow |
| Skip connections | Dashed gray |

## After generation

1. Check topology and labels against the checklist in that file.
2. Fix typos/spacing/colors in Workspace Editor.
3. Export PNG (and SVG if available) to `paper/figures/arch_*.png`.
