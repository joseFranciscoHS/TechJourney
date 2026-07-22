#!/usr/bin/env bash
# Stanford CSD fixel study: export existing arms + train missing Stanford arms.
#
# Self-contained output tree (mirrors rerun_k16_restormer3d_large.sh):
#   tmp/paper_final_k16_stanford_fixels/
#     arrays/<arm>/   denoised volumes for CSD
#     csd/<arm>/      peaks + per-arm metrics
#     proxy/          cross-arm proxy metrics
#     figures/        glyph PNGs
#     registry.jsonl  newly trained Stanford arms
#
# Usage (GPU required for hybrid re-inference / training):
#   bash experiments/rerun_k16_stanford_fixel_arms.sh
#
# Stages (env overrides, default=1):
#   RUN_BASELINES=1     export noisy + MP-PCA + P2S (CPU ok)
#   RUN_EXISTING=1      --skip-train export for DRCNet-3D + Restormer-3D (GPU)
#   RUN_TRAIN_2D=1      train Res-CNN-2D + Restormer-2D on Stanford (GPU)
#   RUN_TRAIN_LARGE=1   train Restormer-3D large on Stanford (GPU)
#   RUN_CSD=1           fit CSD+peaks + proxy metrics (CPU)
#   RUN_FIGURE=1        build glyph figure (CPU; needs ROI scout for final)
#
# Phase-0 only (validate critical path on Restormer-3D):
#   RUN_BASELINES=0 RUN_EXISTING=1 RUN_TRAIN_2D=0 RUN_TRAIN_LARGE=0 \
#   RUN_CSD=1 RUN_FIGURE=0 EXISTING_ARMS=restormer3d \
#     bash experiments/rerun_k16_stanford_fixel_arms.sh
set -euo pipefail

ROOT="${ROOT:-/teamspace/studios/this_studio/TechJourney/DWMRI}"
cd "$ROOT"
source .venv/bin/activate

export RERUN_OUT="${RERUN_OUT:-$PWD/tmp/paper_final_k16_stanford_fixels}"
export RERUN_REGISTRY="$RERUN_OUT/registry.jsonl"
export EXP_ID="paper_final_k16_stanford_fixels"
export ARRAYS_ROOT="$RERUN_OUT/arrays"

RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_EXISTING="${RUN_EXISTING:-1}"
RUN_TRAIN_2D="${RUN_TRAIN_2D:-1}"
RUN_TRAIN_LARGE="${RUN_TRAIN_LARGE:-1}"
RUN_CSD="${RUN_CSD:-1}"
RUN_FIGURE="${RUN_FIGURE:-1}"
EXISTING_ARMS="${EXISTING_ARMS:-drcnet3d,restormer3d}"

JUNE_RERUN="$PWD/tmp/paper_final_k16_rerun_20260628T042410Z"
CKPT_DRCNET="$JUNE_RERUN/drcnet_hybrid_rgs/checkpoints/stanford/b1000/rgs_G150_K16/noise_rician_sigma_0.01/learning_rate_0.00045/stage_3_patch16/best_loss_checkpoint.pth"
CKPT_RESTORMER="$JUNE_RERUN/restormer_hybrid_rgs/checkpoints/stanford/b1000/rgs_G150_K16/noise_rician_sigma_0.01/learning_rate_0.00045/stage_3_patch16/best_loss_checkpoint.pth"

mkdir -p "$RERUN_OUT/runs" "$RERUN_OUT/arrays" "$RERUN_OUT/csd" \
         "$RERUN_OUT/proxy" "$RERUN_OUT/figures" "$RERUN_OUT/rois"

echo "RERUN_OUT=$RERUN_OUT"
echo "RERUN_REGISTRY=$RERUN_REGISTRY"
echo "RUN_BASELINES=$RUN_BASELINES RUN_EXISTING=$RUN_EXISTING RUN_TRAIN_2D=$RUN_TRAIN_2D RUN_TRAIN_LARGE=$RUN_TRAIN_LARGE RUN_CSD=$RUN_CSD RUN_FIGURE=$RUN_FIGURE"

# Shared Stanford K=16 Hybrid RGS protocol (matches June rerun finals).
STANFORD_SHARED=(
  --dataset stanford --regime self_supervised --no-wandb
  --set stanford.train.seed=91022
  --set stanford.train.reproducible=true
  --set stanford.data.shell_sampling_mode=rgs
  --set stanford.data.num_input_volumes=16
  --set stanford.data.shell_gradient_volumes=150
  --set stanford.data.target_channel=15
  --set stanford.reconstruct.mask_p=0.3
  --set stanford.reconstruct.n_preds=23
  --set stanford.reconstruct.n_context_samples=16
  --set stanford.reconstruct.metrics_roi_threshold=0.02
  --set stanford.reconstruct.rescale_to_01=true
  --set stanford.reconstruct.rescale_mode=per_volume
  --set stanford.reconstruct.clip_to_range=true
  --set stanford.reconstruct.compute_dti=false
  --set stanford.reconstruct.save_denoised_npy=true
  --set stanford.reconstruct.save_denoised_nifti=true
  --set "stanford.reconstruct.denoised_out_dir=$ARRAYS_ROOT"
)

LARGE_MODEL_ARGS=(
  --set 'stanford.model.dim=24'
  --set 'stanford.model.num_blocks=[2,4,10]'
  --set 'stanford.model.heads=[1,2,4]'
  --set 'stanford.model.ffn_expansion_factor=2.66'
  --set 'stanford.model.num_refinement_blocks=4'
  # Keep large capacity artifacts off the baseline Restormer-3D path
  # (checkpoints/metrics/images would otherwise collide under rgs_G150_K16/).
  --set 'stanford.model.path_tag=restormer3d_large'
)

# Progressive stages for the large ~2M Restormer (same batch recipe as
# rerun_k16_restormer3d_large.sh on D-Brain). Default targets L4 24GB / L40S.
# Must use single quotes around the JSON so bash does not strip key quotes
# (double-quoted ${VAR:-[...]} mangles "patch_size" and --set gets a raw str).
# T4 (16GB) OOMs at batch=24/patch=32 — override if needed:
#   LARGE_PROGRESSIVE_STAGES='[{"patch_size":32,"batch_size":4,"epochs":300,"step":8},{"patch_size":24,"batch_size":8,"epochs":300,"step":6},{"patch_size":16,"batch_size":16,"epochs":300,"step":4}]' \
#     RUN_TRAIN_LARGE=1 ...
if [[ -z "${LARGE_PROGRESSIVE_STAGES:-}" ]]; then
  LARGE_PROGRESSIVE_STAGES='[{"patch_size":32,"batch_size":24,"epochs":300,"step":8},{"patch_size":24,"batch_size":24,"epochs":300,"step":6},{"patch_size":16,"batch_size":64,"epochs":300,"step":4}]'
fi

cd src

# ---------------------------------------------------------------------------
# Baselines (CPU): noisy + MP-PCA + P2S into arrays/<arm>/
# ---------------------------------------------------------------------------
if [[ "$RUN_BASELINES" == "1" ]]; then
  python -m paper_eval.export_stanford_baseline_arms \
    --arms noisy,mppca,p2s \
    --out-root "$ARRAYS_ROOT" \
    2>&1 | tee "$RERUN_OUT/runs/export_baselines.log"
fi

# ---------------------------------------------------------------------------
# Existing checkpoints (GPU): --skip-train export
# ---------------------------------------------------------------------------
if [[ "$RUN_EXISTING" == "1" ]]; then
  IFS=',' read -r -a _existing <<< "$EXISTING_ARMS"
  for arm in "${_existing[@]}"; do
    case "$arm" in
      drcnet3d)
        python -m drcnet_hybrid_rgs.run "${STANFORD_SHARED[@]}" \
          --skip-train --checkpoint "$CKPT_DRCNET" \
          --exp-id "$EXP_ID" --job-id drcnet3d --recipe stanford_fixel_export \
          --output-root "$RERUN_OUT" --registry-path "$RERUN_REGISTRY" \
          2>&1 | tee "$RERUN_OUT/runs/export_drcnet3d.log"
        ;;
      restormer3d)
        python -m restormer_hybrid_rgs.run "${STANFORD_SHARED[@]}" \
          --skip-train --checkpoint "$CKPT_RESTORMER" \
          --exp-id "$EXP_ID" --job-id restormer3d --recipe stanford_fixel_export \
          --output-root "$RERUN_OUT" --registry-path "$RERUN_REGISTRY" \
          2>&1 | tee "$RERUN_OUT/runs/export_restormer3d.log"
        ;;
      *)
        echo "Unknown EXISTING_ARMS entry: $arm" >&2
        exit 1
        ;;
    esac
  done
fi

# ---------------------------------------------------------------------------
# Train missing Stanford 2D arms (GPU)
# ---------------------------------------------------------------------------
if [[ "$RUN_TRAIN_2D" == "1" ]]; then
  python -m restormer_hybrid_rgs.run2d_hybrid "${STANFORD_SHARED[@]}" \
    --set stanford.model.backbone=res_cnn_2d \
    --exp-id "$EXP_ID" --job-id res_cnn_2d --recipe stanford_fixel_2d \
    --output-root "$RERUN_OUT" --registry-path "$RERUN_REGISTRY" \
    2>&1 | tee "$RERUN_OUT/runs/train_res_cnn_2d.log"

  # Second gate (2D path): assemble->CSD->PNG is covered by RUN_CSD/RUN_FIGURE.
  python -m restormer_hybrid_rgs.run2d_hybrid "${STANFORD_SHARED[@]}" \
    --set stanford.model.backbone=restormer_2d \
    --set stanford.reconstruct.slice_chunk_size=8 \
    --exp-id "$EXP_ID" --job-id restormer2d --recipe stanford_fixel_2d \
    --output-root "$RERUN_OUT" --registry-path "$RERUN_REGISTRY" \
    2>&1 | tee "$RERUN_OUT/runs/train_restormer2d.log"
fi

# ---------------------------------------------------------------------------
# Train missing Restormer-3D large Stanford (GPU)
# ---------------------------------------------------------------------------
if [[ "$RUN_TRAIN_LARGE" == "1" ]]; then
  python -m restormer_hybrid_rgs.run "${STANFORD_SHARED[@]}" "${LARGE_MODEL_ARGS[@]}" \
    --set "stanford.train.progressive.stages=$LARGE_PROGRESSIVE_STAGES" \
    --exp-id "$EXP_ID" --job-id restormer3d_large --recipe stanford_fixel_large \
    --output-root "$RERUN_OUT" --registry-path "$RERUN_REGISTRY" \
    2>&1 | tee "$RERUN_OUT/runs/train_restormer3d_large.log"
fi

cd "$ROOT"

# ---------------------------------------------------------------------------
# CSD + proxy metrics (CPU)
# ---------------------------------------------------------------------------
if [[ "$RUN_CSD" == "1" ]]; then
  python -m paper_eval.csd_fixels \
    --study-root "$RERUN_OUT" \
    2>&1 | tee "$RERUN_OUT/runs/csd_all_arms.log"
fi

# ---------------------------------------------------------------------------
# Glyph figure (CPU; ROI YAML is placeholder until scout)
# ---------------------------------------------------------------------------
if [[ "$RUN_FIGURE" == "1" ]]; then
  cp -f src/paper_eval/stanford_fixel_rois.yaml "$RERUN_OUT/rois/stanford_fixel_rois.yaml"
  python -m paper_eval.plot_fixels \
    --study-root "$RERUN_OUT" \
    --rois "$RERUN_OUT/rois/stanford_fixel_rois.yaml" \
    --out "$RERUN_OUT/figures/fixel_glyphs.png" \
    2>&1 | tee "$RERUN_OUT/runs/plot_fixels.log"
fi

echo "Done. Outputs in: $RERUN_OUT"
