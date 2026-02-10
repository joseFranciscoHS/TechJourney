# DRCNet Hybrid with Transfer Learning

Two-phase pipeline for DWMRI denoising with transfer learning.

## Phase 1: Single-volume training

Train the base `DenoiserNet` on **one** target volume (e.g. `source_volume_index=0`). The model learns to denoise that volume given all others (J-invariance / hybrid MD-S2S). All volumes are in the input; the target volume is Bernoulli-masked; loss is masked MSE on the target.

## Phase 2: Transfer learning

After Phase 1, a **wrapper** is used for each other volume: the trained base is frozen and wrapped with 1×1×1 input and output convolutions. Only these adapter layers are trained per target volume. One wrapper per volume (Option A); reconstruction uses the base for volume 0 and the corresponding wrapper for volumes 1..N-1.

## How to run

From the project root (or with `src` on `PYTHONPATH`):

```python
from src.drcnet_hybrid_tl.run import main

# Full pipeline: Phase 1 training, Phase 2 transfer, then reconstruction
main(dataset="dbrain", train=True, reconstruct=True, transfer_learn=True)
```

Or run the module:

```bash
cd src && python -m drcnet_hybrid_tl.run
```

Default in code is `main(dataset="dbrain")` with `train`, `reconstruct`, and `transfer_learn` all `True`.

## Config

- **Phase 1**: `config.yaml` under `dbrain.train`, `dbrain.model`, `dbrain.data`. Training uses `source_volume_index=0` (fixed in code).
- **Phase 2**: `dbrain.transfer`:
  - `target_volume_indices`: list of volume indices to transfer to (e.g. `[1,2,...,9]`), or `"all"` for 1..num_volumes-1.
  - `num_epochs`, `learning_rate` (typically smaller than Phase 1), `freeze_base: true`, `shared_wrapper: false` (per-volume wrappers).

Reconstruction (Option A) uses the base model for volume 0 and each transfer wrapper for volumes that have a checkpoint under `checkpoint_dir/transfer_vol_{v}/best_loss_checkpoint.pth`.
