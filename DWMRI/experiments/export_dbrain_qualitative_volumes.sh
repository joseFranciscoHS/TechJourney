#!/usr/bin/env bash
# Export D-Brain Hybrid RGS (+ optional MDS2S) full 4D volumes for paper qualitative panels.
#
# GPU required for Hybrid RGS (DRCNet / Restormer). MDS2S can run on CPU.
#
# Usage:
#   bash experiments/export_dbrain_qualitative_volumes.sh
#
# Stages (env overrides, default as noted):
#   RUN_BASELINES=1   stage gt/noisy/mppca/p2s from existing artifacts (CPU)
#   RUN_HYBRID=1      --skip-train export DRCNet + Restormer K=16 σ=0.1 (GPU)
#   RUN_MDS2S=0       MDS2S reconstruct+save (CPU; default off — often already running)
#   HYBRID_ARMS=drcnet3d,restormer3d
#
set -euo pipefail

ROOT="${ROOT:-/teamspace/studios/this_studio/TechJourney/DWMRI}"
cd "$ROOT"
source .venv/bin/activate

export RERUN_OUT="${RERUN_OUT:-$PWD/tmp/paper_final_k16_dbrain_exports}"
export ARRAYS_ROOT="$RERUN_OUT/arrays"
export EXP_ID="paper_final_k16_dbrain_exports"

RUN_BASELINES="${RUN_BASELINES:-1}"
# Hybrid RGS export needs CUDA; default on only when a GPU is visible.
if [[ -z "${RUN_HYBRID:-}" ]]; then
  if python - <<'PY'
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    RUN_HYBRID=1
  else
    RUN_HYBRID=0
    echo "NOTE: no CUDA — defaulting RUN_HYBRID=0 (baselines/MDS2S only)."
  fi
fi
RUN_MDS2S="${RUN_MDS2S:-0}"
HYBRID_ARMS="${HYBRID_ARMS:-drcnet3d,restormer3d}"

JUNE_RERUN="$PWD/tmp/paper_final_k16_rerun_20260628T042410Z"
# Prefer June rerun tree; fall back to paper_final_k16_out.
CKPT_DRCNET="${CKPT_DRCNET:-$JUNE_RERUN/drcnet_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/_subset_f0p6/noise_rician_sigma_0.1/learning_rate_0.00045/best_loss_checkpoint.pth}"
CKPT_RESTORMER="${CKPT_RESTORMER:-$JUNE_RERUN/restormer_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/_subset_f0p6/noise_rician_sigma_0.1/learning_rate_0.00045/best_loss_checkpoint.pth}"
if [[ ! -f "$CKPT_DRCNET" ]]; then
  CKPT_DRCNET="$PWD/tmp/paper_final_k16_out/drcnet_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/_subset_f0p6/noise_rician_sigma_0.1/learning_rate_0.00045/best_loss_checkpoint.pth"
fi
if [[ ! -f "$CKPT_RESTORMER" ]]; then
  CKPT_RESTORMER="$PWD/tmp/paper_final_k16_out/restormer_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/_subset_f0p6/noise_rician_sigma_0.1/learning_rate_0.00045/best_loss_checkpoint.pth"
fi

MDS2S_CKPT="${MDS2S_CKPT:-$PWD/tmp/paper_final_k16_out/mds2s/checkpoints/dbrain/bvalue_2500/num_volumes_60/noise_sigma_0.1/learning_rate_0.0001/best_loss_checkpoint.pth}"

mkdir -p "$RERUN_OUT/runs" "$ARRAYS_ROOT"

echo "RERUN_OUT=$RERUN_OUT"
echo "RUN_BASELINES=$RUN_BASELINES RUN_HYBRID=$RUN_HYBRID RUN_MDS2S=$RUN_MDS2S"
echo "CKPT_DRCNET=$CKPT_DRCNET"
echo "CKPT_RESTORMER=$CKPT_RESTORMER"

# Shared D-Brain K=16 Hybrid RGS protocol (matches June canonical σ=0.1).
DBRAIN_SHARED=(
  --dataset dbrain --regime self_supervised --no-wandb
  --set dbrain.train.seed=91022
  --set dbrain.train.reproducible=true
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
  --set dbrain.reconstruct.compute_dti=false
  --set dbrain.reconstruct.save_denoised_npy=true
  --set dbrain.reconstruct.save_denoised_nifti=true
  --set "dbrain.reconstruct.denoised_out_dir=$ARRAYS_ROOT"
  --output-root "$RERUN_OUT"
)

cd src

if [[ "$RUN_BASELINES" == "1" ]]; then
  python -m paper_eval.stage_dbrain_baseline_arrays \
    --out-root "$ARRAYS_ROOT" \
    2>&1 | tee "$RERUN_OUT/runs/stage_baselines.log"
fi

if [[ "$RUN_HYBRID" == "1" ]]; then
  if ! python - <<'PY'
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    echo "ERROR: CUDA not available. Hybrid RGS D-Brain export needs a GPU." >&2
    echo "Re-run on a GPU machine with RUN_HYBRID=1 (baselines can stay RUN_BASELINES=0)." >&2
    exit 1
  fi
  IFS=',' read -r -a _arms <<< "$HYBRID_ARMS"
  for arm in "${_arms[@]}"; do
    case "$arm" in
      drcnet3d)
        python -m drcnet_hybrid_rgs.run \
          "${DBRAIN_SHARED[@]}" \
          --skip-train --checkpoint "$CKPT_DRCNET" \
          --exp-id "$EXP_ID" --job-id drcnet3d --recipe dbrain_qual_export \
          2>&1 | tee "$RERUN_OUT/runs/export_drcnet3d.log"
        ;;
      restormer3d)
        python -m restormer_hybrid_rgs.run \
          "${DBRAIN_SHARED[@]}" \
          --skip-train --checkpoint "$CKPT_RESTORMER" \
          --exp-id "$EXP_ID" --job-id restormer3d --recipe dbrain_qual_export \
          2>&1 | tee "$RERUN_OUT/runs/export_restormer3d.log"
        ;;
      *)
        echo "Unknown HYBRID arm: $arm" >&2
        exit 1
        ;;
    esac
  done
fi

if [[ "$RUN_MDS2S" == "1" ]]; then
  export MDS2S_SAVE_VOLUME_DIR="$ARRAYS_ROOT/mds2s"
  mkdir -p "$MDS2S_SAVE_VOLUME_DIR"
  # mds2s.run resolves best_loss_checkpoint.pth from its config path layout
  # (src/mds2s/checkpoints/... or PAPER output-root copies). Optional override:
  # copy/symlink MDS2S_CKPT into that resolved path before running.
  if [[ -f "$MDS2S_CKPT" ]]; then
    echo "MDS2S_CKPT=$MDS2S_CKPT (ensure runner path resolves to this file)"
  fi
  python -m mds2s.run \
    --dataset dbrain --skip-train --device cpu --no-wandb --no-images \
    --noise-sigma 0.1 \
    2>&1 | tee "$RERUN_OUT/runs/export_mds2s.log"
fi

echo "Arrays now under $ARRAYS_ROOT:"
ls -la "$ARRAYS_ROOT" || true
echo "Next: python -m paper_eval.plot_dbrain_qualitative --require-hybrids"
