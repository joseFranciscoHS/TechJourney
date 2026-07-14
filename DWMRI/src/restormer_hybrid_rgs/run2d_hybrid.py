"""
Restormer 2D hybrid RGS / sequential-K: config-driven training + reconstruction + metrics/DTI.

Parity target: same crop, K, mask_p, and DTI assembly as ``restormer_hybrid_rgs.run`` (3D),
but convolution runs slice-wise with :class:`Restormer2D`.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from drcnet_hybrid_rgs.data2d import TrainingDataSet2D
from drcnet_hybrid_rgs.reconstruction2d import (
    reconstruct_dwis_rgs_2d,
    reconstruct_dwis_sequential_sliding_k_2d,
)
from paper_eval.dti_metrics import save_dti_metrics, try_compute_dti_errors
from restormer_hybrid_rgs.fit import fit_model
from restormer_hybrid_rgs.model2d import ResCNN2D
from restormer_hybrid_rgs.restormer_arch_2d import Restormer2D
from utils import setup_logging
from utils.data import DBrainDataLoader, invert_normalization
from utils.eval_protocol import (
    apply_reconstruction_eval_protocol,
    compute_roi_mask,
    metrics_policy_dict,
    save_run_manifest,
    summarize_roi,
)
from utils.experiment_runtime import (
    append_registry_line,
    apply_output_root,
    apply_overrides,
    gpu_peak_mem_mb,
    hardware_info,
    losses_dir_from_train_checkpoint_dir,
    now_utc_iso,
)
from utils.metrics import compute_metrics, save_metrics
from utils.repro_seed import (
    configure_cudnn,
    log_runtime_env,
    make_dataloader_generator,
    set_seed,
)
from utils.training_patch_subset import (
    apply_training_patch_subset_from_train_block,
    training_subset_checkpoint_segment,
)
from utils.utils import load_config, noise_path_segment


def _is_rgs(settings) -> bool:
    return getattr(settings.data, "shell_sampling_mode", "sequential") == "rgs"


def _volume_path_segment(settings) -> str:
    mode = getattr(settings.data, "shell_sampling_mode", "sequential")
    g = int(getattr(settings.data, "shell_gradient_volumes", settings.data.num_volumes))
    k = int(getattr(settings.data, "num_input_volumes", settings.model.in_channel))
    return f"{mode}_G{g}_K{k}"


def _backbone_path_segment(settings) -> str:
    """Return a stable filesystem token for the active backbone, normalising aliases."""
    bb = str(getattr(settings.model, "backbone", "res_cnn_2d")).lower()
    if bb in {"res_cnn_2d", "rescnn2d", "cnn"}:
        return "res_cnn_2d"
    if bb in {"restormer_2d", "restormer2d"}:
        return "restormer_2d"
    return bb


def _take_volumes_dwi(settings) -> int:
    mode = getattr(settings.data, "shell_sampling_mode", "sequential")
    if mode in {"rgs", "sequential"}:
        return settings.data.num_b0s + int(
            getattr(settings.data, "shell_gradient_volumes", settings.data.num_volumes)
        )
    return settings.data.num_b0s + settings.data.num_volumes


def _dataset_kwargs(settings):
    mode = getattr(settings.data, "shell_sampling_mode", "sequential")
    kw = {"shell_sampling_mode": mode}
    if mode in {"rgs", "sequential"}:
        kw["num_input_volumes"] = int(
            getattr(settings.data, "num_input_volumes", settings.model.in_channel)
        )
        kw["target_channel"] = int(
            getattr(settings.data, "target_channel", kw["num_input_volumes"] - 1)
        )
    return kw


def main():
    parser = argparse.ArgumentParser(
        description="Restormer hybrid 2D (RGS / sequential-K)"
    )
    parser.add_argument("--dataset", default="dbrain", choices=["dbrain"])
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config key=value (same as restormer_hybrid_rgs.run)",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--registry-path", default=None)
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--recipe", default="restormer_2d_hybrid")
    parser.add_argument("--regime", default="self_supervised")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-reconstruct", action="store_true")
    parser.add_argument(
        "--no-images", action="store_true", help="Unused parity placeholder"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Unused placeholder")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    log_file = setup_logging(logging.INFO)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(script_dir, "config.yaml")
    full_settings = load_config(config_path)
    if args.overrides:
        apply_overrides(full_settings, args.overrides)
    settings = full_settings.dbrain
    apply_output_root(settings, args.output_root)

    train_seed = int(getattr(settings.train, "seed", 42))
    cudnn_fast = not bool(getattr(settings.train, "reproducible", False))
    set_seed(train_seed)
    configure_cudnn(fast=cudnn_fast)
    log_runtime_env(settings.train.device)
    dl_generator = make_dataloader_generator(train_seed)

    data_loader = DBrainDataLoader(
        nii_path=settings.data.nii_path,
        bvecs_path=settings.data.bvecs_path,
        bvalue=settings.data.bvalue,
        noise_sigma=settings.data.noise_sigma,
        noise_type=getattr(settings.data, "noise_type", "rician"),
        n_coils=getattr(settings.data, "noise_n_coils", 1),
    )
    original_data, noisy_data = data_loader.load_data()
    take_volumes = _take_volumes_dwi(settings)
    tx, ty, tz = settings.data.take_x, settings.data.take_y, settings.data.take_z
    original_xyzv_b0 = original_data[:tx, :ty, :tz, :take_volumes]
    noisy_data = noisy_data[:tx, :ty, :tz, settings.data.num_b0s : take_volumes]
    original_data = original_data[:tx, :ty, :tz, settings.data.num_b0s : take_volumes]

    k = int(getattr(settings.data, "num_input_volumes", settings.model.in_channel))
    tc = int(getattr(settings.data, "target_channel", k - 1))
    mode = getattr(settings.data, "shell_sampling_mode", "rgs")

    train_set = TrainingDataSet2D(
        noisy_data,
        shell_sampling_mode=mode,
        num_input_volumes=k,
        target_channel=tc,
        mask_p=settings.train.mask_p,
        patch_hw=(
            int(getattr(settings.data, "patch_2d_h", settings.data.patch_size)),
            int(getattr(settings.data, "patch_2d_w", settings.data.patch_size)),
        ),
        step=int(getattr(settings.data, "patch_2d_step", settings.data.step)),
        sample_rng_seed=train_seed,
    )

    noise_segment = noise_path_segment(
        getattr(settings.data, "noise_type", "rician"),
        getattr(settings.data, "noise_sigma", 0.1),
    )
    bvalue_segment = f"b{getattr(settings.data, 'bvalue', 2500)}"
    vol_seg = _volume_path_segment(settings)
    _sub_seg = training_subset_checkpoint_segment(settings.train)
    _bb_seg = f"_bb_{_backbone_path_segment(settings)}"
    _path_mid = (
        [bvalue_segment, f"2d_{vol_seg}"]
        + ([_sub_seg] if _sub_seg else [])
        + [_bb_seg]
    )
    checkpoint_dir = os.path.join(
        settings.train.checkpoint_dir,
        *_path_mid,
        noise_segment,
        f"learning_rate_{settings.train.learning_rate}",
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    _loss_train_root = losses_dir_from_train_checkpoint_dir(
        settings.train.checkpoint_dir
    )
    loss_dir = os.path.join(
        _loss_train_root,
        *_path_mid,
        noise_segment,
        f"learning_rate_{settings.train.learning_rate}",
    )
    os.makedirs(loss_dir, exist_ok=True)
    metrics_dir = os.path.join(
        settings.reconstruct.metrics_dir,
        *_path_mid,
        noise_segment,
        f"learning_rate_{settings.train.learning_rate}",
    )
    os.makedirs(metrics_dir, exist_ok=True)

    train_set, _n_tot, _n_used = apply_training_patch_subset_from_train_block(
        train_set, settings.train
    )
    train_loader = DataLoader(
        train_set,
        batch_size=settings.train.batch_size,
        shuffle=True,
        generator=dl_generator,
    )

    device = settings.train.device
    backbone = str(getattr(settings.model, "backbone", "res_cnn_2d")).lower()
    dim = getattr(settings.model, "dim", 16)
    ffn = getattr(settings.model, "ffn_expansion_factor", 1.5)
    bias = getattr(settings.model, "bias", False)
    ln_type = getattr(settings.model, "LayerNorm_type", "WithBias")
    num_blocks_cfg = getattr(settings.model, "num_blocks", [1, 2, 2])
    heads_cfg = getattr(settings.model, "heads", [1, 2, 2])

    if backbone in {"res_cnn_2d", "rescnn2d", "cnn"}:
        n_blocks = (
            int(sum(int(v) for v in num_blocks_cfg))
            if isinstance(num_blocks_cfg, (list, tuple))
            else int(num_blocks_cfg)
        )
        model = ResCNN2D(
            inp_channels=k,
            out_channels=1,
            dim=dim,
            num_blocks=max(n_blocks, 1),
            bias=bias,
        ).to(device)
    elif backbone in {"restormer_2d", "restormer2d"}:
        nb = (
            list(num_blocks_cfg)
            if isinstance(num_blocks_cfg, (list, tuple))
            else [1, 2, 2]
        )
        hd = list(heads_cfg) if isinstance(heads_cfg, (list, tuple)) else [1, 2, 2]
        model = Restormer2D(
            inp_channels=k,
            out_channels=1,
            dim=dim,
            num_blocks=nb,
            num_refinement_blocks=int(
                getattr(settings.model, "num_refinement_blocks", 2)
            ),
            heads=hd,
            ffn_expansion_factor=ffn,
            bias=bias,
            LayerNorm_type=ln_type,
            output_activation=getattr(settings.model, "output_activation", "prelu"),
            scale_and_shift=bool(getattr(settings.model, "scale_and_shift", True)),
        ).to(device)
    else:
        raise ValueError(
            f"Unknown model.backbone={backbone!r}; supported: res_cnn_2d | restormer_2d"
        )

    n_params = int(sum(p.numel() for p in model.parameters()))
    logging.info("2D backbone=%s params=%d", backbone, n_params)

    wall_t0 = time.time()
    started = now_utc_iso()
    metrics = metrics_roi = dti_metrics = None
    sec_per_epoch = None
    sec_per_volume = None
    run_status = "success"
    run_error = None

    try:
        if not args.skip_train:
            train_t0 = time.time()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=settings.train.learning_rate
            )
            fit_model(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                train_loader=train_loader,
                num_epochs=int(getattr(settings.train, "num_epochs", 5)),
                device=device,
                checkpoint_dir=checkpoint_dir,
                loss_dir=loss_dir,
                use_amp=getattr(settings.train, "use_amp", True),
                supervised_mode=bool(getattr(settings.train, "supervised", False)),
                cudnn_fast=cudnn_fast,
            )
            sec_per_epoch = float(time.time() - train_t0) / float(
                max(int(getattr(settings.train, "num_epochs", 1)), 1)
            )

        if not args.skip_reconstruct:
            ckpt = args.checkpoint or os.path.join(
                checkpoint_dir, "best_loss_checkpoint.pth"
            )
            if os.path.isfile(ckpt):
                try:
                    bundle = torch.load(ckpt, map_location=device, weights_only=False)
                except TypeError:
                    bundle = torch.load(ckpt, map_location=device)
                model.load_state_dict(bundle["model_state_dict"], strict=False)

            noisy_v = np.transpose(noisy_data.astype(np.float32), (3, 0, 1, 2))
            infer_t0 = time.time()
            _pred_chunk = getattr(settings.reconstruct, "pred_chunk_size", None)
            _slice_chunk = getattr(settings.reconstruct, "slice_chunk_size", None)
            if _is_rgs(settings):
                recon_vxyz = reconstruct_dwis_rgs_2d(
                    model,
                    noisy_v,
                    device,
                    mask_p=settings.reconstruct.mask_p,
                    n_preds=int(settings.reconstruct.n_preds),
                    n_context=int(
                        getattr(settings.reconstruct, "n_context_samples", 4)
                    ),
                    target_channel=tc,
                    num_input=k,
                    seed=train_seed,
                    pred_chunk_size=_pred_chunk,
                    slice_chunk_size=_slice_chunk,
                )
            else:
                recon_vxyz = reconstruct_dwis_sequential_sliding_k_2d(
                    model,
                    noisy_v,
                    device,
                    mask_p=settings.reconstruct.mask_p,
                    n_preds=int(settings.reconstruct.n_preds),
                    num_input=k,
                    target_channel=tc,
                    seed=train_seed,
                    pred_chunk_size=_pred_chunk,
                    slice_chunk_size=_slice_chunk,
                )
            recon_xyzv = np.transpose(recon_vxyz, (1, 2, 3, 0))
            sec_per_volume = float(time.time() - infer_t0) / float(recon_xyzv.shape[-1])
            recon_xyzv = apply_reconstruction_eval_protocol(
                recon_xyzv,
                original_data,
                rescale_to_01=bool(
                    getattr(settings.reconstruct, "rescale_to_01", False)
                ),
                rescale_mode=str(
                    getattr(settings.reconstruct, "rescale_mode", "per_volume")
                ),
                clip_to_range=bool(
                    getattr(settings.reconstruct, "clip_to_range", False)
                ),
            )

            os.makedirs(metrics_dir, exist_ok=True)
            metrics = compute_metrics(original_data, recon_xyzv)
            save_metrics(metrics, metrics_dir, filename="metrics.json")
            roi_thr = getattr(settings.reconstruct, "metrics_roi_threshold", 0.02)
            roi_mask = compute_roi_mask(original_data, roi_thr)
            if roi_mask is not None:
                n_roi, roi_pct = summarize_roi(roi_mask)
                logging.info(
                    "ROI mask for 2D Restormer: %s voxels (%.1f%%)",
                    n_roi,
                    roi_pct,
                )
                metrics_roi = compute_metrics(original_data, recon_xyzv, mask=roi_mask)
                save_metrics(metrics_roi, metrics_dir, filename="metrics_roi.json")

            if getattr(settings.reconstruct, "compute_dti", True):
                try:
                    gtab = data_loader.load_gradient_table()
                    bvals = np.asarray(gtab.bvals)[:take_volumes]
                    bvecs = np.asarray(gtab.bvecs)[:take_volumes]
                    nb0 = int(settings.data.num_b0s)
                    gt_xyzv = original_xyzv_b0.astype(np.float64)
                    den_xyzv = np.concatenate(
                        [gt_xyzv[..., :nb0], recon_xyzv.astype(np.float64)],
                        axis=-1,
                    )
                    norm_params = getattr(data_loader, "norm_params_", None)
                    if norm_params is not None:
                        gt_xyzv = invert_normalization(
                            gt_xyzv, norm_params[:take_volumes]
                        )
                        den_dwis = invert_normalization(
                            recon_xyzv.astype(np.float64),
                            norm_params[nb0:take_volumes],
                        )
                        den_xyzv = np.concatenate(
                            [gt_xyzv[..., :nb0], den_dwis.astype(np.float64)],
                            axis=-1,
                        )
                    dti_metrics = try_compute_dti_errors(
                        den_xyzv,
                        gt_xyzv,
                        bvals,
                        bvecs,
                        roi_threshold=roi_thr,
                    )
                except Exception as exc:  # pragma: no cover - runtime dependent
                    dti_metrics = {
                        "fa_mae": None,
                        "md_mae": None,
                        "ad_mae": None,
                        "rd_mae": None,
                        "dti_reference": "clean_gt",
                        "dti_skipped_reason": str(exc),
                    }
            else:
                dti_metrics = {
                    "fa_mae": None,
                    "md_mae": None,
                    "ad_mae": None,
                    "rd_mae": None,
                    "dti_reference": "clean_gt",
                    "dti_skipped_reason": "compute_dti_false",
                }
            save_dti_metrics(dti_metrics, metrics_dir)

            save_run_manifest(
                out_dir=metrics_dir,
                seed=train_seed,
                reproducible=bool(getattr(settings.train, "reproducible", False)),
                runtime_device=str(device),
                config={
                    "dataset": "dbrain",
                    "architecture": "restormer_hybrid_rgs",
                    "sampling_mode": mode,
                    "k_input": int(k),
                    "g_shell": int(
                        getattr(
                            settings.data,
                            "shell_gradient_volumes",
                            settings.data.num_volumes,
                        )
                    ),
                    "n_context_samples": int(
                        getattr(settings.reconstruct, "n_context_samples", 0)
                    ),
                    "n_preds": int(getattr(settings.reconstruct, "n_preds", 0)),
                    "dimensionality": "2d",
                    "backbone": backbone,
                },
                metrics_policy=metrics_policy_dict(
                    reference_name="clean_gt",
                    rescale_to_01=bool(
                        getattr(settings.reconstruct, "rescale_to_01", False)
                    ),
                    rescale_mode=str(
                        getattr(settings.reconstruct, "rescale_mode", "per_volume")
                    ),
                    clip_to_range=bool(
                        getattr(settings.reconstruct, "clip_to_range", False)
                    ),
                    roi_threshold=roi_thr,
                ),
            )

    except Exception as exc:
        run_status = "failed"
        run_error = str(exc)
        raise
    finally:
        payload = {
            "schema_version": "v1",
            "exp_id": args.exp_id,
            "job_id": args.job_id,
            "recipe": args.recipe,
            "status": run_status,
            "error": run_error,
            "timestamps": {"start_utc": started, "end_utc": now_utc_iso()},
            "duration_s": time.time() - wall_t0,
            "stage": "train_reconstruct"
            if (not args.skip_train and not args.skip_reconstruct)
            else ("train" if not args.skip_train else "reconstruct"),
            "dataset": "dbrain",
            "regime": args.regime,
            "architecture": "restormer",
            "dimensionality": "2d",
            "backbone": backbone,
            "sampling_mode": mode,
            "sampling_config": {
                "g_shell": int(
                    getattr(
                        settings.data,
                        "shell_gradient_volumes",
                        settings.data.num_volumes,
                    )
                ),
                "k_input": int(k),
                "target_channel": int(tc),
                "window_policy": "sliding_last_target",
            },
            "inference_config": {
                "n_context_samples": int(
                    getattr(settings.reconstruct, "n_context_samples", 0)
                ),
                "n_preds": int(getattr(settings.reconstruct, "n_preds", 0)),
            },
            "train_config": {
                "epochs": int(getattr(settings.train, "num_epochs", 0)),
                "batch_size": int(getattr(settings.train, "batch_size", 0)),
                "lr": float(getattr(settings.train, "learning_rate", 0.0)),
                "progressive_enabled": bool(
                    getattr(
                        getattr(settings.train, "progressive", {}), "enabled", False
                    )
                ),
            },
            "control_metrics": {
                "n_params": int(n_params),
                "sec_per_epoch": sec_per_epoch,
                "sec_per_volume": sec_per_volume,
                "peak_gpu_mem_mb": gpu_peak_mem_mb(device),
            },
            "quality_metrics_full": metrics,
            "quality_metrics_roi": metrics_roi,
            "dti_metrics": dti_metrics,
            "hardware": hardware_info(device),
        }
        append_registry_line(args.registry_path, payload)
        logging.info("Restormer 2D hybrid run complete. log=%s", log_file)


if __name__ == "__main__":
    main()
