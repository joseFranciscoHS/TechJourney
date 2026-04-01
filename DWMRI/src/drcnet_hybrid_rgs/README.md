# DRCNet-Hybrid (RGS): Gated Denoising for DWMRI

This package implements **DRCNet** with **Scheme 2** hybrid (spatial Bernoulli mask + multi-gradient context) self-supervised denoising. It extends the classic **sequential** hybrid with optional **RGS–Hybrid** (**R**andom **G**radient **S**ubset) training and matching inference.

> **Note:** The research plan referred to this line of work as `drcnet_hybrid_3`. The implementation package name is **`drcnet_hybrid_rgs`** (RGS = Random Gradient Subset). Use `python -m drcnet_hybrid_rgs.run`; do not rename the folder without updating imports.

## Modes: `sequential` vs `rgs`

| `data.shell_sampling_mode` | Training input | Inference |
|-----------------------------|----------------|-----------|
| **`sequential`** (default) | Every forward uses **all V** loaded DWI channels; target rotates over volume index (classic hybrid). | `reconstruct_dwis`: full `(V,…)` stack; mask one volume; spatial MC (`n_preds`). |
| **`rgs`** | **G** = full shell (e.g. 60). Each sample draws **K** distinct gradient indices **without replacement**, **order = draw order**; Bernoulli mask + loss on fixed channel `target_channel` (default **9** = 10th slot). | `reconstruct_dwis_rgs`: for each target index **k**, Monte Carlo over **random (K−1)-tuples** from the other gradients, volume **k** fixed at `target_channel`; inner spatial `n_preds` per context; average over `n_context_samples`. |

**Config keys (under `data`):**

- `shell_sampling_mode`: `sequential` | `rgs`
- `num_input_volumes`: **K** (must match `model.in_channel`)
- `shell_gradient_volumes`: **G** when `mode=rgs` — controls `take_volumes` slice after b0s (DBrain `run.py`)
- `target_channel`: 0-based index for mask + loss (typically **K−1**)

**Reconstruction (under `reconstruct`):**

- `n_context_samples`: outer MC draws for RGS (random contexts)
- `n_preds`: inner spatial Bernoulli passes (same as sequential)

**Complexity (RGS inference):** proportional to **V × n_context_samples × n_preds × forward** (expensive; tune for speed).

Checkpoint / loss paths use segments like `rgs_G60_K10` when RGS is enabled.

## Progressive Learning (Optional)

Training can use an optional **progressive learning** strategy: multiple stages with increasing patch size and step. When enabled, `num_epochs` and `batch_size` at the top level are overridden per stage.

- **Config**: In `config.yaml` under `train`, set `progressive.enabled: true` and define `progressive.stages` (list of `patch_size`, `batch_size`, `epochs`, `step` per stage).
- **Behavior**: Each stage builds a new dataset with that stage’s patch size and step, a new optimizer and scheduler, and writes checkpoints/losses to stage-specific subdirs. After the last stage, the best checkpoint is copied to `checkpoint_dir/best_loss_checkpoint.pth` for reconstruction.
- **RGS**: The **channel** dimension of each patch is **K** (`num_input_volumes`), not **G**.

Example (in `config.yaml` under e.g. `dbrain.train`):

```yaml
progressive:
  enabled: false
  stages:
    - patch_size: 8
      batch_size: 64
      epochs: 10
      step: 2
```

Optional: set `train.seed` for reproducible subset sampling across runs.

## Running (from `DWMRI/src` with `PYTHONPATH=.`)

```bash
python -m drcnet_hybrid_rgs.run
python -m drcnet_hybrid_rgs.run_stanford_fewvol
```

## References

- [`J_invariance_DWMRI_denoising_report.md`](../J_invariance_DWMRI_denoising_report.md) — Scheme 2
- Discussion: [`discussion/threads/`](../discussion/threads/)
