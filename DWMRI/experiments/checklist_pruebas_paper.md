# Checklist operativo de pruebas para paper (ejecucion 1 a 1)

Este checklist traduce el plan de `src/plan_para_escribir_el_paper.md` a ejecucion practica y enumera **cada** `job_id` de `experiments/paper_manifest_final.yaml`.
Todos los comandos se corren desde la raiz de `DWMRI/`.

---

## 0) Setup rapido (una sola vez)

- [ ] Instalar entorno y deps
  - `uv sync --extra dev && source .venv/bin/activate`
- [ ] Definir directorio de salida (debe coincidir con rutas del manifiesto si usas los defaults)
  - `export PAPER_OUT="$PWD/tmp/paper_final_out"`
- [ ] Definir registry
  - `export PAPER_REGISTRY=$PAPER_OUT/registry.jsonl`
- [ ] Directorio compartido export GT/noisy (mismo `--out-dir` que `export_dbrain_npy_final` en el YAML)
  - `export PAPER_SHARED_NPY="$PWD/tmp/paper_final_shared_npy"`
- [ ] Reanudar solo jobs pendientes del manifiesto completo (salta los ya exitosos en `driver_state.json`)
  - `python experiments/driver.py --manifest experiments/paper_manifest_final.yaml --exp-id paper_final --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY" --resume --retry-failed --fail-fast`

---

## 1) Baselines y modelos principales (tabla principal)

### 1.1 Export par GT/noisy para MP-PCA

- [x] `job_id: export_dbrain_npy_final`
  - `python experiments/paper_export_dbrain_volume_pair.py --config src/drcnet_hybrid_rgs/config.yaml --out-dir "$PAPER_SHARED_NPY"`

### 1.2 Baselines clasicos

- [x] `job_id: mppca_dbrain_final`
  - `python -m paper_eval.baselines.mppca_run --noisy "$PAPER_SHARED_NPY/noisy_dwi_xyzv.npy" --gt "$PAPER_SHARED_NPY/gt_dwi_xyzv.npy" --out-dir "$PAPER_OUT/baselines/mppca/dbrain" --metrics-roi-threshold 0.02 --rescale-mode per_volume`
- [x] `job_id: p2s_dbrain_dipy_final`
  - `python -m p2s.run --dataset dbrain --backend dipy --seed 91021 --reproducible true --no-wandb`
- [x] `job_id: p2s_dbrain_sklearn_reference_final`
  - `python -m p2s.run --dataset dbrain --backend sklearn_reference --seed 91021 --reproducible true --no-wandb`
- [ ] `job_id: mds2s_dbrain_final`
  - `python -m mds2s.run --dataset dbrain --seed 91021 --reproducible true --no-wandb`

### 1.3 Metodo propuesto (self-supervised)

- [ ] `job_id: drcnet_dbrain_rgs_final`
  - `python -m drcnet_hybrid_rgs.run --dataset dbrain --no-wandb --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.data.shell_sampling_mode=rgs --set dbrain.data.num_input_volumes=16 --set dbrain.data.shell_gradient_volumes=60 --set dbrain.data.target_channel=15 --set dbrain.reconstruct.metrics_roi_threshold=0.02 --set dbrain.reconstruct.rescale_to_01=true --set dbrain.reconstruct.rescale_mode=per_volume --set dbrain.reconstruct.clip_to_range=true --set dbrain.reconstruct.compute_dti=true --exp-id paper_final --job-id drcnet_dbrain_rgs_final --recipe drcnet_main --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: restormer_dbrain_rgs_final`
  - `python -m restormer_hybrid_rgs.run --dataset dbrain --no-wandb --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.data.shell_sampling_mode=rgs --set dbrain.data.num_input_volumes=16 --set dbrain.data.shell_gradient_volumes=60 --set dbrain.data.target_channel=15 --set dbrain.reconstruct.metrics_roi_threshold=0.02 --set dbrain.reconstruct.rescale_to_01=true --set dbrain.reconstruct.rescale_mode=per_volume --set dbrain.reconstruct.clip_to_range=true --set dbrain.reconstruct.compute_dti=true --exp-id paper_final --job-id restormer_dbrain_rgs_final --recipe restormer_main --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`

### 1.4 Cota superior supervisada

- [ ] `job_id: drcnet_dbrain_supervised_upperbound_final`
  - `python -m drcnet_hybrid_rgs.run --dataset dbrain --regime supervised --no-wandb --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.train.supervised=true --set dbrain.train.checkpoint_dir=drcnet_hybrid_rgs/checkpoints/dbrain_supervised --set dbrain.data.shell_sampling_mode=rgs --set dbrain.data.num_input_volumes=16 --set dbrain.data.shell_gradient_volumes=60 --set dbrain.data.target_channel=15 --set dbrain.reconstruct.metrics_roi_threshold=0.02 --set dbrain.reconstruct.rescale_to_01=true --set dbrain.reconstruct.rescale_mode=per_volume --set dbrain.reconstruct.clip_to_range=true --set dbrain.reconstruct.metrics_dir=drcnet_hybrid_rgs/metrics/dbrain_supervised --set dbrain.reconstruct.images_dir=drcnet_hybrid_rgs/images/dbrain_supervised --set dbrain.reconstruct.compute_dti=true --exp-id paper_final --job-id drcnet_dbrain_supervised_upperbound_final --recipe supervised_upperbound --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: restormer_dbrain_supervised_upperbound_final`
  - `python -m restormer_hybrid_rgs.run --dataset dbrain --regime supervised --no-wandb --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.train.supervised=true --set dbrain.train.checkpoint_dir=restormer_hybrid_rgs/checkpoints/dbrain_supervised --set dbrain.data.shell_sampling_mode=rgs --set dbrain.data.num_input_volumes=16 --set dbrain.data.shell_gradient_volumes=60 --set dbrain.data.target_channel=15 --set dbrain.reconstruct.metrics_roi_threshold=0.02 --set dbrain.reconstruct.rescale_to_01=true --set dbrain.reconstruct.rescale_mode=per_volume --set dbrain.reconstruct.clip_to_range=true --set dbrain.reconstruct.metrics_dir=restormer_hybrid_rgs/metrics/dbrain_supervised --set dbrain.reconstruct.images_dir=restormer_hybrid_rgs/images/dbrain_supervised --set dbrain.reconstruct.compute_dti=true --exp-id paper_final --job-id restormer_dbrain_supervised_upperbound_final --recipe supervised_upperbound --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`

---

## 2) Ablaciones clave (D-Brain)

### 2.1 Sampling: Sequential vs RGS (K=16)

- [ ] `job_id: drcnet_dbrain_seq_k16_ablation`
  - `python -m drcnet_hybrid_rgs.run --dataset dbrain --no-wandb --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.data.shell_sampling_mode=sequential --set dbrain.data.num_input_volumes=16 --set dbrain.data.shell_gradient_volumes=60 --set dbrain.data.target_channel=15 --set dbrain.reconstruct.metrics_roi_threshold=0.02 --set dbrain.reconstruct.rescale_to_01=true --set dbrain.reconstruct.rescale_mode=per_volume --set dbrain.reconstruct.clip_to_range=true --set dbrain.reconstruct.compute_dti=true --exp-id paper_final --job-id drcnet_dbrain_seq_k16_ablation --recipe sampling_sequential_vs_rgs --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: restormer_dbrain_seq_k16_ablation`
  - `python -m restormer_hybrid_rgs.run --dataset dbrain --no-wandb --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.data.shell_sampling_mode=sequential --set dbrain.data.num_input_volumes=16 --set dbrain.data.shell_gradient_volumes=60 --set dbrain.data.target_channel=15 --set dbrain.reconstruct.metrics_roi_threshold=0.02 --set dbrain.reconstruct.rescale_to_01=true --set dbrain.reconstruct.rescale_mode=per_volume --set dbrain.reconstruct.clip_to_range=true --set dbrain.reconstruct.compute_dti=true --exp-id paper_final --job-id restormer_dbrain_seq_k16_ablation --recipe sampling_sequential_vs_rgs --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`

### 2.2 K sweep D-Brain (check individual por k)

- [ ] `job_id: drcnet_dbrain_k5_ablation` (`k=5`)
- [ ] `job_id: drcnet_dbrain_k10_ablation` (`k=10`)
- [ ] `job_id: drcnet_dbrain_k16_ablation` (`k=16`)
- [ ] `job_id: drcnet_dbrain_k24_ablation` (`k=24`)
- [ ] `job_id: drcnet_dbrain_k30_ablation` (`k=30`)
- [ ] `job_id: restormer_dbrain_k5_ablation` (`k=5`)
- [ ] `job_id: restormer_dbrain_k10_ablation` (`k=10`)
- [ ] `job_id: restormer_dbrain_k16_ablation` (`k=16`)
- [ ] `job_id: restormer_dbrain_k24_ablation` (`k=24`)
- [ ] `job_id: restormer_dbrain_k30_ablation` (`k=30`)
  - Comando de corrida (reanuda y avanza job por job en orden del manifest):
  - `python experiments/driver.py --manifest experiments/paper_manifest_final.yaml --exp-id paper_final --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY" --resume --retry-failed --fail-fast`

### 2.3 3D vs 2D (ablacion critica)

- [ ] `job_id: drcnet_dbrain_2d_rgs_k16_ablation`
  - `python -m drcnet_hybrid_rgs.run2d_hybrid --dataset dbrain --regime self_supervised --no-wandb --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.data.shell_sampling_mode=rgs --set dbrain.data.num_input_volumes=16 --set dbrain.data.shell_gradient_volumes=60 --set dbrain.data.target_channel=15 --set dbrain.reconstruct.mask_p=0.3 --set dbrain.reconstruct.n_preds=12 --set dbrain.reconstruct.n_context_samples=16 --set dbrain.reconstruct.metrics_roi_threshold=0.02 --set dbrain.reconstruct.rescale_to_01=true --set dbrain.reconstruct.rescale_mode=per_volume --set dbrain.reconstruct.clip_to_range=true --set dbrain.reconstruct.compute_dti=true --exp-id paper_final --job-id drcnet_dbrain_2d_rgs_k16_ablation --recipe conv2d_vs_conv3d --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: restormer_dbrain_2d_rgs_k16_ablation`
  - `python -m restormer_hybrid_rgs.run2d_hybrid --dataset dbrain --regime self_supervised --no-wandb --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.data.shell_sampling_mode=rgs --set dbrain.data.num_input_volumes=16 --set dbrain.data.shell_gradient_volumes=60 --set dbrain.data.target_channel=15 --set dbrain.reconstruct.mask_p=0.3 --set dbrain.reconstruct.n_preds=5 --set dbrain.reconstruct.n_context_samples=16 --set dbrain.reconstruct.metrics_roi_threshold=0.02 --set dbrain.reconstruct.rescale_to_01=true --set dbrain.reconstruct.rescale_mode=per_volume --set dbrain.reconstruct.clip_to_range=true --set dbrain.reconstruct.compute_dti=true --exp-id paper_final --job-id restormer_dbrain_2d_rgs_k16_ablation --recipe conv2d_vs_conv3d --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`

### 2.4 Entrenamiento progresivo vs estandar (plan 1.7)

- [ ] `job_id: drcnet_dbrain_progressive_off_ablation`
- [ ] `job_id: restormer_dbrain_progressive_off_ablation`
  - Comando de corrida:
  - `python experiments/driver.py --manifest experiments/paper_manifest_final.yaml --exp-id paper_final --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY" --resume --retry-failed --fail-fast`

### 2.5 Sensibilidad a mask_p D-Brain (check individual por valor)

- [ ] `job_id: drcnet_dbrain_maskp_01_ablation` (`mask_p=0.1`)
- [ ] `job_id: drcnet_dbrain_maskp_02_ablation` (`mask_p=0.2`)
- [ ] `job_id: drcnet_dbrain_maskp_03_ablation` (`mask_p=0.3`)
- [ ] `job_id: drcnet_dbrain_maskp_05_ablation` (`mask_p=0.5`)
- [ ] `job_id: drcnet_dbrain_maskp_07_ablation` (`mask_p=0.7`)
- [ ] `job_id: restormer_dbrain_maskp_01_ablation` (`mask_p=0.1`)
- [ ] `job_id: restormer_dbrain_maskp_02_ablation` (`mask_p=0.2`)
- [ ] `job_id: restormer_dbrain_maskp_03_ablation` (`mask_p=0.3`)
- [ ] `job_id: restormer_dbrain_maskp_05_ablation` (`mask_p=0.5`)
- [ ] `job_id: restormer_dbrain_maskp_07_ablation` (`mask_p=0.7`)
  - Comando de corrida:
  - `python experiments/driver.py --manifest experiments/paper_manifest_final.yaml --exp-id paper_final --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY" --resume --retry-failed --fail-fast`

### 2.6 Sensibilidad a n_context D-Brain (check individual por valor)

- [ ] `job_id: drcnet_dbrain_ncontext_1_ablation` (`n_context=1`)
- [ ] `job_id: drcnet_dbrain_ncontext_3_ablation` (`n_context=3`)
- [ ] `job_id: drcnet_dbrain_ncontext_5_ablation` (`n_context=5`)
- [ ] `job_id: drcnet_dbrain_ncontext_12_ablation` (`n_context=12`)
- [ ] `job_id: drcnet_dbrain_ncontext_24_ablation` (`n_context=24`)
- [ ] `job_id: drcnet_dbrain_ncontext_48_ablation` (`n_context=48`)
- [ ] `job_id: restormer_dbrain_ncontext_1_ablation` (`n_context=1`)
- [ ] `job_id: restormer_dbrain_ncontext_3_ablation` (`n_context=3`)
- [ ] `job_id: restormer_dbrain_ncontext_5_ablation` (`n_context=5`)
- [ ] `job_id: restormer_dbrain_ncontext_12_ablation` (`n_context=12`)
- [ ] `job_id: restormer_dbrain_ncontext_24_ablation` (`n_context=24`)
- [ ] `job_id: restormer_dbrain_ncontext_48_ablation` (`n_context=48`)
  - Comando de corrida:
  - `python experiments/driver.py --manifest experiments/paper_manifest_final.yaml --exp-id paper_final --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY" --resume --retry-failed --fail-fast`

### 2.7 Sensibilidad a n_preds D-Brain (check individual por valor)

- [ ] `job_id: drcnet_dbrain_npreds_1_ablation` (`n_preds=1`)
- [ ] `job_id: drcnet_dbrain_npreds_3_ablation` (`n_preds=3`)
- [ ] `job_id: drcnet_dbrain_npreds_5_ablation` (`n_preds=5`)
- [ ] `job_id: drcnet_dbrain_npreds_10_ablation` (`n_preds=10`)
- [ ] `job_id: drcnet_dbrain_npreds_15_ablation` (`n_preds=15`)
- [ ] `job_id: drcnet_dbrain_npreds_20_ablation` (`n_preds=20`)
- [ ] `job_id: restormer_dbrain_npreds_1_ablation` (`n_preds=1`)
- [ ] `job_id: restormer_dbrain_npreds_3_ablation` (`n_preds=3`)
- [ ] `job_id: restormer_dbrain_npreds_5_ablation` (`n_preds=5`)
- [ ] `job_id: restormer_dbrain_npreds_10_ablation` (`n_preds=10`)
- [ ] `job_id: restormer_dbrain_npreds_15_ablation` (`n_preds=15`)
- [ ] `job_id: restormer_dbrain_npreds_20_ablation` (`n_preds=20`)
  - Comando de corrida:
  - `python experiments/driver.py --manifest experiments/paper_manifest_final.yaml --exp-id paper_final --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY" --resume --retry-failed --fail-fast`

### 2.8 Inferencia y analisis D-Brain (plan seccion 5; requiere checkpoints K16 sigma 0.1)

- [ ] `job_id: drcnet_dbrain_arch_compare_parity_final`
  - `python -m drcnet_hybrid_rgs.run --dataset dbrain --no-wandb --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.data.shell_sampling_mode=rgs --set dbrain.data.num_input_volumes=16 --set dbrain.data.shell_gradient_volumes=60 --set dbrain.data.target_channel=15 --set dbrain.reconstruct.n_context_samples=24 --set dbrain.reconstruct.n_preds=10 --set dbrain.reconstruct.metrics_roi_threshold=0.02 --set dbrain.reconstruct.rescale_to_01=true --set dbrain.reconstruct.rescale_mode=per_volume --set dbrain.reconstruct.clip_to_range=true --set dbrain.reconstruct.compute_dti=true --exp-id paper_final --job-id drcnet_dbrain_arch_compare_parity_final --recipe architecture_parity --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: restormer_dbrain_arch_compare_parity_final`
  - `python -m restormer_hybrid_rgs.run --dataset dbrain --no-wandb --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.data.shell_sampling_mode=rgs --set dbrain.data.num_input_volumes=16 --set dbrain.data.shell_gradient_volumes=60 --set dbrain.data.target_channel=15 --set dbrain.reconstruct.n_context_samples=24 --set dbrain.reconstruct.n_preds=10 --set dbrain.reconstruct.metrics_roi_threshold=0.02 --set dbrain.reconstruct.rescale_to_01=true --set dbrain.reconstruct.rescale_mode=per_volume --set dbrain.reconstruct.clip_to_range=true --set dbrain.reconstruct.compute_dti=true --exp-id paper_final --job-id restormer_dbrain_arch_compare_parity_final --recipe architecture_parity --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: drcnet_dbrain_ncontext24_seed_91031_infer`
  - `python -m drcnet_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint drcnet_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00045/best_loss_checkpoint.pth --no-wandb --set dbrain.train.seed=91031 --set dbrain.train.reproducible=true --set dbrain.reconstruct.n_context_samples=24 --set dbrain.reconstruct.n_preds=10 --exp-id paper_final --job-id drcnet_dbrain_ncontext24_seed_91031_infer --recipe n_context_stability --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: drcnet_dbrain_ncontext24_seed_91041_infer`
  - `python -m drcnet_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint drcnet_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00045/best_loss_checkpoint.pth --no-wandb --set dbrain.train.seed=91041 --set dbrain.train.reproducible=true --set dbrain.reconstruct.n_context_samples=24 --set dbrain.reconstruct.n_preds=10 --exp-id paper_final --job-id drcnet_dbrain_ncontext24_seed_91041_infer --recipe n_context_stability --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: restormer_dbrain_ncontext24_seed_91031_infer`
  - `python -m restormer_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint restormer_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00044/best_loss_checkpoint.pth --no-wandb --set dbrain.train.seed=91031 --set dbrain.train.reproducible=true --set dbrain.reconstruct.n_context_samples=24 --set dbrain.reconstruct.n_preds=10 --exp-id paper_final --job-id restormer_dbrain_ncontext24_seed_91031_infer --recipe n_context_stability --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: restormer_dbrain_ncontext24_seed_91041_infer`
  - `python -m restormer_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint restormer_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00044/best_loss_checkpoint.pth --no-wandb --set dbrain.train.seed=91041 --set dbrain.train.reproducible=true --set dbrain.reconstruct.n_context_samples=24 --set dbrain.reconstruct.n_preds=10 --exp-id paper_final --job-id restormer_dbrain_ncontext24_seed_91041_infer --recipe n_context_stability --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: drcnet_dbrain_gap_standard_infer`
  - `python -m drcnet_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint drcnet_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00045/best_loss_checkpoint.pth --no-wandb --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.reconstruct.n_context_samples=24 --set dbrain.reconstruct.n_preds=10 --set dbrain.reconstruct.mask_p=0.3 --exp-id paper_final --job-id drcnet_dbrain_gap_standard_infer --recipe training_inference_gap_proxy --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: drcnet_dbrain_gap_masked_proxy_infer`
  - `python -m drcnet_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint drcnet_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00045/best_loss_checkpoint.pth --no-wandb --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.reconstruct.n_context_samples=24 --set dbrain.reconstruct.n_preds=10 --set dbrain.reconstruct.mask_p=1.0 --exp-id paper_final --job-id drcnet_dbrain_gap_masked_proxy_infer --recipe training_inference_gap_proxy --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: restormer_dbrain_gap_standard_infer`
  - `python -m restormer_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint restormer_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00044/best_loss_checkpoint.pth --no-wandb --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.reconstruct.n_context_samples=24 --set dbrain.reconstruct.n_preds=10 --set dbrain.reconstruct.mask_p=0.3 --exp-id paper_final --job-id restormer_dbrain_gap_standard_infer --recipe training_inference_gap_proxy --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: restormer_dbrain_gap_masked_proxy_infer`
  - `python -m restormer_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint restormer_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00044/best_loss_checkpoint.pth --no-wandb --set dbrain.train.seed=91021 --set dbrain.train.reproducible=true --set dbrain.reconstruct.n_context_samples=24 --set dbrain.reconstruct.n_preds=10 --set dbrain.reconstruct.mask_p=1.0 --exp-id paper_final --job-id restormer_dbrain_gap_masked_proxy_infer --recipe training_inference_gap_proxy --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: drcnet_dbrain_inference_time_ctx5_pred5`
  - `python -m drcnet_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint drcnet_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00045/best_loss_checkpoint.pth --no-wandb --set dbrain.reconstruct.n_context_samples=5 --set dbrain.reconstruct.n_preds=5 --exp-id paper_final --job-id drcnet_dbrain_inference_time_ctx5_pred5 --recipe inference_time_grid --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: drcnet_dbrain_inference_time_ctx12_pred10`
  - `python -m drcnet_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint drcnet_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00045/best_loss_checkpoint.pth --no-wandb --set dbrain.reconstruct.n_context_samples=12 --set dbrain.reconstruct.n_preds=10 --exp-id paper_final --job-id drcnet_dbrain_inference_time_ctx12_pred10 --recipe inference_time_grid --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: drcnet_dbrain_inference_time_ctx24_pred12`
  - `python -m drcnet_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint drcnet_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00045/best_loss_checkpoint.pth --no-wandb --set dbrain.reconstruct.n_context_samples=24 --set dbrain.reconstruct.n_preds=12 --exp-id paper_final --job-id drcnet_dbrain_inference_time_ctx24_pred12 --recipe inference_time_grid --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: drcnet_dbrain_inference_time_ctx48_pred20`
  - `python -m drcnet_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint drcnet_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00045/best_loss_checkpoint.pth --no-wandb --set dbrain.reconstruct.n_context_samples=48 --set dbrain.reconstruct.n_preds=20 --exp-id paper_final --job-id drcnet_dbrain_inference_time_ctx48_pred20 --recipe inference_time_grid --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: restormer_dbrain_inference_time_ctx5_pred5`
  - `python -m restormer_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint restormer_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00044/best_loss_checkpoint.pth --no-wandb --set dbrain.reconstruct.n_context_samples=5 --set dbrain.reconstruct.n_preds=5 --exp-id paper_final --job-id restormer_dbrain_inference_time_ctx5_pred5 --recipe inference_time_grid --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: restormer_dbrain_inference_time_ctx12_pred10`
  - `python -m restormer_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint restormer_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00044/best_loss_checkpoint.pth --no-wandb --set dbrain.reconstruct.n_context_samples=12 --set dbrain.reconstruct.n_preds=10 --exp-id paper_final --job-id restormer_dbrain_inference_time_ctx12_pred10 --recipe inference_time_grid --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: restormer_dbrain_inference_time_ctx24_pred12`
  - `python -m restormer_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint restormer_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00044/best_loss_checkpoint.pth --no-wandb --set dbrain.reconstruct.n_context_samples=24 --set dbrain.reconstruct.n_preds=12 --exp-id paper_final --job-id restormer_dbrain_inference_time_ctx24_pred12 --recipe inference_time_grid --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: restormer_dbrain_inference_time_ctx48_pred20`
  - `python -m restormer_hybrid_rgs.run --dataset dbrain --skip-train --checkpoint restormer_hybrid_rgs/checkpoints/dbrain/b2500/rgs_G60_K16/noise_rician_sigma_0.1/learning_rate_0.00044/best_loss_checkpoint.pth --no-wandb --set dbrain.reconstruct.n_context_samples=48 --set dbrain.reconstruct.n_preds=20 --exp-id paper_final --job-id restormer_dbrain_inference_time_ctx48_pred20 --recipe inference_time_grid --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`

---

## 3) Ruido sintetico: sigma sweep (D-Brain only, hay GT limpio)

Los `export_*_sigma_*` del YAML escriben bajo `$PWD/tmp/paper_final_shared_npy_sigma_*` (rutas fijas en el manifiesto).

### 3.1 Sigma 0.05

- [ ] `job_id: export_dbrain_npy_sigma_050_final`
- [ ] `job_id: mppca_dbrain_sigma_050_final`
- [ ] `job_id: p2s_dbrain_dipy_sigma_050_final`
- [ ] `job_id: p2s_dbrain_sklearn_reference_sigma_050_final`
- [ ] `job_id: mds2s_dbrain_sigma_050_final`
- [ ] `job_id: drcnet_dbrain_sigma_050_ablation`
- [ ] `job_id: restormer_dbrain_sigma_050_ablation`
  - Comando de corrida:
  - `python experiments/driver.py --manifest experiments/paper_manifest_final.yaml --exp-id paper_final --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY" --resume --retry-failed --fail-fast`

### 3.2 Sigma 0.10

- [ ] `job_id: export_dbrain_npy_sigma_100_final`
- [ ] `job_id: mppca_dbrain_sigma_100_final`
- [ ] `job_id: p2s_dbrain_dipy_sigma_100_final`
- [ ] `job_id: p2s_dbrain_sklearn_reference_sigma_100_final`
- [ ] `job_id: mds2s_dbrain_sigma_100_final`
- [ ] `job_id: drcnet_dbrain_sigma_100_ablation`
- [ ] `job_id: restormer_dbrain_sigma_100_ablation`
  - Comando de corrida:
  - `python experiments/driver.py --manifest experiments/paper_manifest_final.yaml --exp-id paper_final --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY" --resume --retry-failed --fail-fast`

### 3.3 Sigma 0.15

- [ ] `job_id: export_dbrain_npy_sigma_150_final`
- [ ] `job_id: mppca_dbrain_sigma_150_final`
- [ ] `job_id: p2s_dbrain_dipy_sigma_150_final`
- [ ] `job_id: p2s_dbrain_sklearn_reference_sigma_150_final`
- [ ] `job_id: mds2s_dbrain_sigma_150_final`
- [ ] `job_id: drcnet_dbrain_sigma_150_ablation`
- [ ] `job_id: restormer_dbrain_sigma_150_ablation`
  - Comando de corrida:
  - `python experiments/driver.py --manifest experiments/paper_manifest_final.yaml --exp-id paper_final --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY" --resume --retry-failed --fail-fast`

### 3.4 Sigma 0.20

- [ ] `job_id: export_dbrain_npy_sigma_200_final`
- [ ] `job_id: mppca_dbrain_sigma_200_final`
- [ ] `job_id: p2s_dbrain_dipy_sigma_200_final`
- [ ] `job_id: p2s_dbrain_sklearn_reference_sigma_200_final`
- [ ] `job_id: mds2s_dbrain_sigma_200_final`
- [ ] `job_id: drcnet_dbrain_sigma_200_ablation`
- [ ] `job_id: restormer_dbrain_sigma_200_ablation`
  - Comando de corrida:
  - `python experiments/driver.py --manifest experiments/paper_manifest_final.yaml --exp-id paper_final --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY" --resume --retry-failed --fail-fast`

---

## 4) Stanford (generalizacion sin GT)

### 4.1 Runs principales y baselines

- [ ] `job_id: drcnet_stanford_rgs_final`
  - `python -m drcnet_hybrid_rgs.run --dataset stanford --no-wandb --set stanford.train.seed=91022 --set stanford.train.reproducible=true --set stanford.data.shell_sampling_mode=rgs --set stanford.data.num_input_volumes=16 --set stanford.data.shell_gradient_volumes=150 --set stanford.data.target_channel=15 --set stanford.reconstruct.metrics_roi_threshold=0.02 --set stanford.reconstruct.rescale_to_01=true --set stanford.reconstruct.rescale_mode=per_volume --set stanford.reconstruct.clip_to_range=true --set stanford.reconstruct.compute_dti=true --exp-id paper_final --job-id drcnet_stanford_rgs_final --recipe drcnet_main --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: restormer_stanford_rgs_final`
  - `python -m restormer_hybrid_rgs.run --dataset stanford --no-wandb --set stanford.train.seed=91022 --set stanford.train.reproducible=true --set stanford.data.shell_sampling_mode=rgs --set stanford.data.num_input_volumes=16 --set stanford.data.shell_gradient_volumes=150 --set stanford.data.target_channel=15 --set stanford.reconstruct.metrics_roi_threshold=0.02 --set stanford.reconstruct.rescale_to_01=true --set stanford.reconstruct.rescale_mode=per_volume --set stanford.reconstruct.clip_to_range=true --set stanford.reconstruct.compute_dti=true --exp-id paper_final --job-id restormer_stanford_rgs_final --recipe restormer_main --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY"`
- [ ] `job_id: p2s_stanford_dipy_final`
  - `python -m p2s.run --dataset stanford --backend dipy --seed 91022 --reproducible true --no-wandb`
- [ ] `job_id: mds2s_stanford_final`
  - `python -m mds2s.run --dataset stanford --seed 91022 --reproducible true --no-wandb`

### 4.2 K sweep Stanford (`k=5,10,16,24,30`)

- [ ] `job_id: drcnet_stanford_k5_ablation`
- [ ] `job_id: drcnet_stanford_k10_ablation`
- [ ] `job_id: drcnet_stanford_k16_ablation`
- [ ] `job_id: drcnet_stanford_k24_ablation`
- [ ] `job_id: drcnet_stanford_k30_ablation`
- [ ] `job_id: restormer_stanford_k5_ablation`
- [ ] `job_id: restormer_stanford_k10_ablation`
- [ ] `job_id: restormer_stanford_k16_ablation`
- [ ] `job_id: restormer_stanford_k24_ablation`
- [ ] `job_id: restormer_stanford_k30_ablation`
  - Comando de corrida:
  - `python experiments/driver.py --manifest experiments/paper_manifest_final.yaml --exp-id paper_final --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY" --resume --retry-failed --fail-fast`

### 4.3 Sensibilidad a mask_p Stanford

- [ ] `job_id: drcnet_stanford_maskp_01_ablation`
- [ ] `job_id: drcnet_stanford_maskp_02_ablation`
- [ ] `job_id: drcnet_stanford_maskp_03_ablation`
- [ ] `job_id: drcnet_stanford_maskp_05_ablation`
- [ ] `job_id: drcnet_stanford_maskp_07_ablation`
- [ ] `job_id: restormer_stanford_maskp_01_ablation`
- [ ] `job_id: restormer_stanford_maskp_02_ablation`
- [ ] `job_id: restormer_stanford_maskp_03_ablation`
- [ ] `job_id: restormer_stanford_maskp_05_ablation`
- [ ] `job_id: restormer_stanford_maskp_07_ablation`
  - Comando de corrida:
  - `python experiments/driver.py --manifest experiments/paper_manifest_final.yaml --exp-id paper_final --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY" --resume --retry-failed --fail-fast`

### 4.4 Sensibilidad a n_context Stanford

- [ ] `job_id: drcnet_stanford_ncontext_1_ablation`
- [ ] `job_id: drcnet_stanford_ncontext_3_ablation`
- [ ] `job_id: drcnet_stanford_ncontext_5_ablation`
- [ ] `job_id: drcnet_stanford_ncontext_12_ablation`
- [ ] `job_id: drcnet_stanford_ncontext_24_ablation`
- [ ] `job_id: drcnet_stanford_ncontext_48_ablation`
- [ ] `job_id: restormer_stanford_ncontext_1_ablation`
- [ ] `job_id: restormer_stanford_ncontext_3_ablation`
- [ ] `job_id: restormer_stanford_ncontext_5_ablation`
- [ ] `job_id: restormer_stanford_ncontext_12_ablation`
- [ ] `job_id: restormer_stanford_ncontext_24_ablation`
- [ ] `job_id: restormer_stanford_ncontext_48_ablation`
  - Comando de corrida:
  - `python experiments/driver.py --manifest experiments/paper_manifest_final.yaml --exp-id paper_final --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY" --resume --retry-failed --fail-fast`

---

## 5) Cierre de artefactos para tablas del paper

- [ ] `job_id: summarize_registry_final`
  - `python -m paper_eval.summarize_registry --registry "$PAPER_REGISTRY" --out "$PAPER_OUT/paper_tables/registry_summary.csv"`
- [ ] `job_id: collect_paper_artifacts_final`
  - `python experiments/collect_paper_artifacts.py --output-root "$PAPER_OUT" --registry "$PAPER_REGISTRY" --out-dir "$PAPER_OUT/paper_tables"`

---

## 6) Comando unico de continuacion (reanuda donde se quedo)

Si prefieres no correr comando por comando y dejar que avance en orden del manifest:

- [ ] Reanudar pipeline completo
  - `python experiments/driver.py --manifest experiments/paper_manifest_final.yaml --exp-id paper_final --output-root "$PAPER_OUT" --registry-path "$PAPER_REGISTRY" --resume --retry-failed --fail-fast`
