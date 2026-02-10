# DWMRI

Diffusion-weighted MRI reconstruction and denoising.

## drcnet_hybrid_tl

DRCNet hybrid learning with **transfer learning**: train on a single volume (Phase 1), then adapt to other volumes via lightweight 1×1×1 input/output adapters (Phase 2). See [src/drcnet_hybrid_tl/README.md](src/drcnet_hybrid_tl/README.md) for usage.
