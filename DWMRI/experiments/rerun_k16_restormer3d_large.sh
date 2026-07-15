#!/usr/bin/env bash
# Larger-backbone 3D Restormer ablation on D-Brain.
# Validates the effect of scaling Restormer-3D from ~0.18M to ~2M params (depth + FFN)
# under the same Hybrid RGS protocol. Mirrors rerun_k16_restormer2d_ablation.sh:
# self-contained output directory with its own registry and paper_tables.
#
# Usage (Lightning AI studio layout):
#   bash experiments/rerun_k16_restormer3d_large.sh
#
# Quick memory/time calibration only (no long run):
#   CALIBRATE=1 bash experiments/rerun_k16_restormer3d_large.sh
#
# Skip the dim=12 baseline arm (reuse existing headline instead):
#   INCLUDE_BASELINE=0 bash experiments/rerun_k16_restormer3d_large.sh
#
# Override output root:
#   RERUN_OUT=/path/to/out bash experiments/rerun_k16_restormer3d_large.sh
set -euo pipefail

ROOT="${ROOT:-/teamspace/studios/this_studio/TechJourney/DWMRI}"
cd "$ROOT"
source .venv/bin/activate

export RERUN_OUT="${RERUN_OUT:-$PWD/tmp/paper_final_k16_restormer3d_large}"
export RERUN_REGISTRY="$RERUN_OUT/registry.jsonl"
export EXP_ID="paper_final_k16_restormer3d_large"

CALIBRATE="${CALIBRATE:-0}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"

mkdir -p "$RERUN_OUT/runs" "$RERUN_OUT/paper_tables"

echo "RERUN_OUT=$RERUN_OUT"
echo "RERUN_REGISTRY=$RERUN_REGISTRY"
echo "CALIBRATE=$CALIBRATE  INCLUDE_BASELINE=$INCLUDE_BASELINE"

# ---------------------------------------------------------------------------
# Shared D-Brain protocol flags (identical to rerun_k16_base.sh / 2d ablation)
# ---------------------------------------------------------------------------
SHARED_ARGS=(
  --dataset dbrain --regime self_supervised --no-wandb
  --set dbrain.train.seed=91021
  --set dbrain.train.reproducible=true
  --set dbrain.train.subset_fraction=0.6
  --set dbrain.data.shell_sampling_mode=rgs
  --set dbrain.data.num_input_volumes=16
  --set dbrain.data.shell_gradient_volumes=60
  --set dbrain.data.target_channel=15
  --set dbrain.reconstruct.mask_p=0.3
  --set dbrain.reconstruct.n_preds=12
  --set dbrain.reconstruct.n_context_samples=16
  --set dbrain.reconstruct.metrics_roi_threshold=0.02
  --set dbrain.reconstruct.rescale_to_01=true
  --set dbrain.reconstruct.rescale_mode=per_volume
  --set dbrain.reconstruct.clip_to_range=true
  --set dbrain.reconstruct.compute_dti=true
)

# Large ~2M-param backbone: depth + FFN growth (see plan Step 2).
# dim=24, channels per level = 24/48/96 -> heads 1/2/4 divide cleanly.
# Requires list-literal --set parsing (utils.experiment_runtime.parse_override_value).
LARGE_MODEL_ARGS=(
  --set 'dbrain.model.dim=24'
  --set 'dbrain.model.num_blocks=[2,4,10]'
  --set 'dbrain.model.heads=[1,2,4]'
  --set 'dbrain.model.ffn_expansion_factor=2.66'
  --set 'dbrain.model.num_refinement_blocks=4'
)

cd src

# ---------------------------------------------------------------------------
# Calibration: short probe to confirm exact param count, peak memory, and
# per-epoch / per-volume timing for the large arm, then stop.
# ---------------------------------------------------------------------------
if [[ "$CALIBRATE" == "1" ]]; then
  CALIB_REGISTRY="$RERUN_OUT/calib_registry.jsonl"
  echo "Running CALIBRATION only -> $CALIB_REGISTRY"
  python -m restormer_hybrid_rgs.run "${SHARED_ARGS[@]}" "${LARGE_MODEL_ARGS[@]}" \
    --set dbrain.train.progressive.enabled=false \
    --set dbrain.train.num_epochs=2 \
    --set dbrain.train.subset_fraction=0.05 \
    --set dbrain.reconstruct.n_preds=1 \
    --exp-id "$EXP_ID" \
    --job-id restormer_dbrain_3d_large2M_k16_calib \
    --recipe capacity_scaling_3d \
    --output-root "$RERUN_OUT" \
    --registry-path "$CALIB_REGISTRY" \
    2>&1 | tee "$RERUN_OUT/runs/restormer_dbrain_3d_large2M_k16_calib.log"
  echo "Calibration done. Inspect control_metrics (n_params, sec_per_epoch, sec_per_volume, peak_gpu_mem_mb):"
  echo "  $CALIB_REGISTRY"
  echo "Full-run time estimate ~ sec_per_epoch*360 + sec_per_volume*60."
  exit 0
fi

# ---------------------------------------------------------------------------
# Arm A: baseline Restormer-3D (dim=12, default config) - apples-to-apples
# reference. Skip with INCLUDE_BASELINE=0 (existing headline in
# tmp/paper_final_k16_rerun_20260628T042410Z/registry.jsonl is equivalent).
# ---------------------------------------------------------------------------
if [[ "$INCLUDE_BASELINE" == "1" ]]; then
  python -m restormer_hybrid_rgs.run "${SHARED_ARGS[@]}" \
    --exp-id "$EXP_ID" \
    --job-id restormer_dbrain_3d_baseline_k16 \
    --recipe capacity_scaling_3d \
    --output-root "$RERUN_OUT" \
    --registry-path "$RERUN_REGISTRY" \
    2>&1 | tee "$RERUN_OUT/runs/restormer_dbrain_3d_baseline_k16.log"
fi

# ---------------------------------------------------------------------------
# Arm B: large ~2M Restormer-3D (depth + FFN growth)
# OOM fallback (not needed on 48 GB; only if ported to a 24 GB card): lower the
# per-stage batch_size values in the progressive block of
# src/restormer_hybrid_rgs/config.yaml (dbrain.train.progressive.stages).
# ---------------------------------------------------------------------------
python -m restormer_hybrid_rgs.run "${SHARED_ARGS[@]}" "${LARGE_MODEL_ARGS[@]}" \
  --exp-id "$EXP_ID" \
  --job-id restormer_dbrain_3d_large2M_k16 \
  --recipe capacity_scaling_3d \
  --output-root "$RERUN_OUT" \
  --registry-path "$RERUN_REGISTRY" \
  2>&1 | tee "$RERUN_OUT/runs/restormer_dbrain_3d_large2M_k16.log"

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
