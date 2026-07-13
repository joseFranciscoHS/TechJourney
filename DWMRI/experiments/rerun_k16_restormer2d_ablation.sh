#!/usr/bin/env bash
# Rerun the D-Brain 2D ablation with both Res-CNN-2D and capacity-matched Restormer-2D.
# Mirrors the structure of rerun_k16_base.sh; produces a self-contained output directory
# with its own registry and paper_tables for the conv2d_vs_conv3d ablation.
#
# Usage (Lightning AI studio layout):
#   bash experiments/rerun_k16_restormer2d_ablation.sh
#
# Override output root:
#   RERUN_OUT=/path/to/out bash experiments/rerun_k16_restormer2d_ablation.sh
set -euo pipefail

ROOT="${ROOT:-/teamspace/studios/this_studio/TechJourney/DWMRI}"
cd "$ROOT"
source .venv/bin/activate

export RERUN_OUT="${RERUN_OUT:-$PWD/tmp/paper_final_k16_restormer2d_ablation}"
export RERUN_REGISTRY="$RERUN_OUT/registry.jsonl"
export EXP_ID="paper_final_k16_restormer2d"

mkdir -p "$RERUN_OUT/runs" "$RERUN_OUT/paper_tables"

echo "RERUN_OUT=$RERUN_OUT"
echo "RERUN_REGISTRY=$RERUN_REGISTRY"

cd src

# ---------------------------------------------------------------------------
# Arm A: lightweight Res-CNN-2D (default backbone — existing ablation arm)
# ---------------------------------------------------------------------------
python -m restormer_hybrid_rgs.run2d_hybrid --dataset dbrain --regime self_supervised --no-wandb \
  --set dbrain.train.seed=91021 \
  --set dbrain.train.reproducible=true \
  --set dbrain.train.subset_fraction=0.6 \
  --set dbrain.data.shell_sampling_mode=rgs \
  --set dbrain.data.num_input_volumes=16 \
  --set dbrain.data.shell_gradient_volumes=60 \
  --set dbrain.data.target_channel=15 \
  --set dbrain.reconstruct.mask_p=0.3 \
  --set dbrain.reconstruct.n_preds=5 \
  --set dbrain.reconstruct.n_context_samples=16 \
  --set dbrain.reconstruct.metrics_roi_threshold=0.02 \
  --set dbrain.reconstruct.rescale_to_01=true \
  --set dbrain.reconstruct.rescale_mode=per_volume \
  --set dbrain.reconstruct.clip_to_range=true \
  --set dbrain.reconstruct.compute_dti=true \
  --exp-id "$EXP_ID" \
  --job-id restormer_dbrain_2d_rgs_k16_ablation \
  --recipe conv2d_vs_conv3d \
  --output-root "$RERUN_OUT" \
  --registry-path "$RERUN_REGISTRY" \
  2>&1 | tee "$RERUN_OUT/runs/restormer_dbrain_2d_rgs_k16_ablation.log"

# ---------------------------------------------------------------------------
# Arm B: capacity-matched Restormer-2D (new backbone)
# ---------------------------------------------------------------------------
python -m restormer_hybrid_rgs.run2d_hybrid --dataset dbrain --regime self_supervised --no-wandb \
  --set dbrain.model.backbone=restormer_2d \
  --set dbrain.reconstruct.slice_chunk_size=8 \
  --set dbrain.train.seed=91021 \
  --set dbrain.train.reproducible=true \
  --set dbrain.train.subset_fraction=0.6 \
  --set dbrain.data.shell_sampling_mode=rgs \
  --set dbrain.data.num_input_volumes=16 \
  --set dbrain.data.shell_gradient_volumes=60 \
  --set dbrain.data.target_channel=15 \
  --set dbrain.reconstruct.mask_p=0.3 \
  --set dbrain.reconstruct.n_preds=5 \
  --set dbrain.reconstruct.n_context_samples=16 \
  --set dbrain.reconstruct.metrics_roi_threshold=0.02 \
  --set dbrain.reconstruct.rescale_to_01=true \
  --set dbrain.reconstruct.rescale_mode=per_volume \
  --set dbrain.reconstruct.clip_to_range=true \
  --set dbrain.reconstruct.compute_dti=true \
  --exp-id "$EXP_ID" \
  --job-id restormer_dbrain_restormer2d_rgs_k16_ablation \
  --recipe conv2d_vs_conv3d \
  --output-root "$RERUN_OUT" \
  --registry-path "$RERUN_REGISTRY" \
  2>&1 | tee "$RERUN_OUT/runs/restormer_dbrain_restormer2d_rgs_k16_ablation.log"

# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------
cd "$ROOT"

python -m paper_eval.summarize_registry \
  --registry "$RERUN_REGISTRY" \
  --out "$RERUN_OUT/paper_tables/registry_summary.csv"

python experiments/collect_paper_artifacts.py \
  --output-root "$RERUN_OUT" \
  --registry "$RERUN_REGISTRY" \
  --out-dir "$RERUN_OUT/paper_tables"

echo "Done. Outputs in: $RERUN_OUT"
