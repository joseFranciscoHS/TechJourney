# DRCNet-Hybrid: Gated Denoising for DWMRI

This module implements **DRCNet** with **hybrid Multi-Directional Self2Self (MD-S2S)** training for J-invariant denoising of diffusion-weighted MRI.

## Progressive Learning (Optional)

Training can use an optional **progressive learning** strategy: multiple stages with increasing patch size and step. When enabled, `num_epochs` and `batch_size` at the top level are overridden per stage.

- **Config**: In `config.yaml` under `train`, set `progressive.enabled: true` and define `progressive.stages` (list of `patch_size`, `batch_size`, `epochs`, `step` per stage).
- **Behavior**: Each stage builds a new dataset with that stage’s patch size and step, a new optimizer and scheduler, and writes checkpoints/losses to stage-specific subdirs. After the last stage, the best checkpoint is copied to `checkpoint_dir/best_loss_checkpoint.pth` for reconstruction.
- **Default**: `progressive.enabled: false`; standard single-dataset training is unchanged.

Example (in `config.yaml` under e.g. `dbrain.train`):

```yaml
progressive:
  enabled: false
  stages:
    - patch_size: 8
      batch_size: 64
      epochs: 10
      step: 2
    - patch_size: 12
      batch_size: 32
      epochs: 10
      step: 4
    - patch_size: 16
      batch_size: 16
      epochs: 10
      step: 4
```

Optional: set `train.seed` for reproducible subset sampling across runs.
