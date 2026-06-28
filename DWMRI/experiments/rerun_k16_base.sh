#!/usr/bin/env bash
set -euo pipefail

cd /teamspace/studios/this_studio/TechJourney/DWMRI
source .venv/bin/activate

export RERUN_OUT="${RERUN_OUT:-$PWD/tmp/paper_final_k16_rerun_20260628T042410Z}"
export RERUN_REGISTRY="$RERUN_OUT/registry.jsonl"
export EXP_ID="paper_final_k16_rerun"

mkdir -p "$RERUN_OUT/runs" "$RERUN_OUT/paper_tables"

echo "RERUN_OUT=$RERUN_OUT"
echo "RERUN_REGISTRY=$RERUN_REGISTRY"

cd src

python -m drcnet_hybrid_rgs.run --dataset dbrain --no-wandb \
  --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.train.subset_fraction=0.6 \
  --set dbrain.data.shell_sampling_mode=rgs --set dbrain.data.num_input_volumes=16 --set dbrain.data.shell_gradient_volumes=60 --set dbrain.data.target_channel=15 \
  --set dbrain.reconstruct.metrics_roi_threshold=0.02 --set dbrain.reconstruct.rescale_to_01=true --set dbrain.reconstruct.rescale_mode=per_volume --set dbrain.reconstruct.clip_to_range=true --set dbrain.reconstruct.compute_dti=true \
  --exp-id "$EXP_ID" --job-id drcnet_dbrain_rgs_final --recipe drcnet_main \
  --output-root "$RERUN_OUT" --registry-path "$RERUN_REGISTRY" \
  2>&1 | tee "$RERUN_OUT/runs/drcnet_dbrain_rgs_final.log"

python -m restormer_hybrid_rgs.run --dataset dbrain --no-wandb \
  --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.train.subset_fraction=0.6 \
  --set dbrain.data.shell_sampling_mode=rgs --set dbrain.data.num_input_volumes=16 --set dbrain.data.shell_gradient_volumes=60 --set dbrain.data.target_channel=15 \
  --set dbrain.reconstruct.metrics_roi_threshold=0.02 --set dbrain.reconstruct.rescale_to_01=true --set dbrain.reconstruct.rescale_mode=per_volume --set dbrain.reconstruct.clip_to_range=true --set dbrain.reconstruct.compute_dti=true \
  --exp-id "$EXP_ID" --job-id restormer_dbrain_rgs_final --recipe restormer_main \
  --output-root "$RERUN_OUT" --registry-path "$RERUN_REGISTRY" \
  2>&1 | tee "$RERUN_OUT/runs/restormer_dbrain_rgs_final.log"

python -m drcnet_hybrid_rgs.run --dataset stanford --no-wandb \
  --set stanford.train.seed=91022 --set stanford.train.reproducible=true \
  --set stanford.data.shell_sampling_mode=rgs --set stanford.data.num_input_volumes=16 --set stanford.data.shell_gradient_volumes=150 --set stanford.data.target_channel=15 \
  --set stanford.reconstruct.metrics_roi_threshold=0.02 --set stanford.reconstruct.rescale_to_01=true --set stanford.reconstruct.rescale_mode=per_volume --set stanford.reconstruct.clip_to_range=true --set stanford.reconstruct.compute_dti=false \
  --exp-id "$EXP_ID" --job-id drcnet_stanford_rgs_final --recipe drcnet_main \
  --output-root "$RERUN_OUT" --registry-path "$RERUN_REGISTRY" \
  2>&1 | tee "$RERUN_OUT/runs/drcnet_stanford_rgs_final.log"

python -m restormer_hybrid_rgs.run --dataset stanford --no-wandb \
  --set stanford.train.seed=91022 --set stanford.train.reproducible=true \
  --set stanford.data.shell_sampling_mode=rgs --set stanford.data.num_input_volumes=16 --set stanford.data.shell_gradient_volumes=150 --set stanford.data.target_channel=15 \
  --set stanford.reconstruct.metrics_roi_threshold=0.02 --set stanford.reconstruct.rescale_to_01=true --set stanford.reconstruct.rescale_mode=per_volume --set stanford.reconstruct.clip_to_range=true --set stanford.reconstruct.compute_dti=false \
  --exp-id "$EXP_ID" --job-id restormer_stanford_rgs_final --recipe restormer_main \
  --output-root "$RERUN_OUT" --registry-path "$RERUN_REGISTRY" \
  2>&1 | tee "$RERUN_OUT/runs/restormer_stanford_rgs_final.log"

cd /teamspace/studios/this_studio/TechJourney/DWMRI

python -m paper_eval.summarize_registry \
  --registry "$RERUN_REGISTRY" \
  --out "$RERUN_OUT/paper_tables/registry_summary.csv"

python experiments/collect_paper_artifacts.py \
  --output-root "$RERUN_OUT" \
  --registry "$RERUN_REGISTRY" \
  --out-dir "$RERUN_OUT/paper_tables"

echo "Done. Outputs in: $RERUN_OUT"