# DRCNet Hybrid – One Model per Volume

Train one **independent** `DenoiserNet` per b-value volume. No transfer learning: each model learns to denoise a single target volume only (J-invariance / hybrid MD-S2S). For 10 volumes you get 10 trained models.

## Training

- One shared **dataset** is built once (sliding windows over the 4D DWMRI). Before each volume’s training run, `set_source_volume_index(i)` is called so that every sample uses volume `i` as the target.
- The target volume is **Bernoulli-masked** in the input; the model predicts the target; loss is masked MSE on the target (same as hybrid MD-S2S).
- For each volume index `i` in `0 .. num_volumes-1`: a **new** `DenoiserNet` is created, trained on that volume only, and saved under `checkpoint_dir/volume_{i}/` (e.g. `best_loss_checkpoint.pth`).

## Reconstruction

- One model per volume: for each volume index, the corresponding checkpoint is loaded (`volume_{i}/best_loss_checkpoint.pth`), then `reconstruct_single_volume` is run. Results are concatenated into the full DWI and used for metrics and visualizations.

## How to run

From the project root (or with `src` on `PYTHONPATH`):

```python
from src.drcnet_hybrid_multiple_networks.run import main

# Train one model per volume, then reconstruct
main(dataset="dbrain", train=True, reconstruct=True)
```

Or run the module:

```bash
cd src && python -m drcnet_hybrid_multiple_networks.run
```

Default is `main(dataset="dbrain")` with `train=True`, `reconstruct=True`, `generate_images=True`. The `transfer_learn` argument is ignored (no transfer in this pipeline).

## Config

- **config.yaml** under `dbrain.train`, `dbrain.model`, `dbrain.data` (no `transfer` section).
- Checkpoints and losses live under `drcnet_hybrid_multiple_networks/checkpoints/...` and `drcnet_hybrid_multiple_networks/losses/...`, with one subdir per volume: `volume_0`, `volume_1`, …, `volume_{N-1}`.
- Reconstruction expects `checkpoint_dir/volume_{i}/best_loss_checkpoint.pth` for each volume `i`.
