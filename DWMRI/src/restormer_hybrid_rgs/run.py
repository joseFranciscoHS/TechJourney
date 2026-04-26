import gc
import argparse
import logging
import os
import shutil
import time

import numpy as np
import torch
import wandb
from restormer_hybrid_rgs.data import TrainingDataSet
from restormer_hybrid_rgs.fit import fit_model
from restormer_hybrid_rgs.model import Restormer3D
from restormer_hybrid_rgs.reconstruction import (
    reconstruct_dwis,
    reconstruct_dwis_rgs,
    reconstruct_dwis_sequential_sliding_k,
)
from torch.utils.data import DataLoader, Subset
from utils import setup_logging
from utils.checkpoint import load_checkpoint
from utils.data import (
    DBrainDataLoader,
    StanfordDataLoader,
    compute_brain_mask,
    rescale_reconstruction_to_01,
)
from utils.metrics import (
    compute_metrics,
    fully_compare_volumes,
    save_metrics,
    visualize_single_volume,
)
from utils.experiment_runtime import (
    append_registry_line,
    apply_output_root,
    apply_overrides,
    gpu_peak_mem_mb,
    hardware_info,
    now_utc_iso,
)
from utils.multi_gpu import create_multi_gpu_config_from_dict, setup_multi_gpu
from utils.utils import load_config, noise_path_segment


def _is_rgs(settings) -> bool:
    return getattr(settings.data, "shell_sampling_mode", "sequential") == "rgs"


def _is_sequential(settings) -> bool:
    return getattr(settings.data, "shell_sampling_mode", "sequential") == "sequential"


def _volume_path_segment(settings) -> str:
    if _is_rgs(settings):
        g = int(getattr(settings.data, "shell_gradient_volumes", settings.data.num_volumes))
        k = int(getattr(settings.data, "num_input_volumes", settings.model.in_channel))
        return f"rgs_G{g}_K{k}"
    if _is_sequential(settings):
        g = int(getattr(settings.data, "shell_gradient_volumes", settings.data.num_volumes))
        k = int(getattr(settings.data, "num_input_volumes", settings.model.in_channel))
        return f"sequential_G{g}_K{k}"
    return f"num_volumes_{settings.data.num_volumes}"


def _patch_volume_dim(settings) -> int:
    if _is_rgs(settings) or _is_sequential(settings):
        return int(getattr(settings.data, "num_input_volumes", settings.model.in_channel))
    return settings.data.num_volumes


def _take_volumes_dwi(settings) -> int:
    if _is_rgs(settings) or _is_sequential(settings):
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
        kw["target_channel"] = int(getattr(settings.data, "target_channel", 9))
    return kw


def fit_progressive(
    model,
    settings,
    noisy_data,
    original_data,
    brain_mask,
    checkpoint_dir,
    loss_dir,
    patch_filter_method,
    min_signal_threshold,
):
    """
    Train model progressively with increasing patch sizes.

    Progressive learning stages use per-stage patch_size, step, batch_size, and
    epochs; each stage gets a new dataset, optimizer, and scheduler.
    """
    stages = settings.train.progressive.stages
    total_stages = len(stages)
    subset_seed = getattr(settings.train, "seed", 42)
    use_amp = getattr(settings.train, "use_amp", True)

    logging.info(
        f"Progressive Learning: {total_stages} stages, AMP={'enabled' if use_amp else 'disabled'}"
    )
    for i, stage in enumerate(stages):
        logging.info(
            f"  Stage {i+1}: patch={stage.patch_size}³, batch={stage.batch_size}, "
            f"epochs={stage.epochs}, step={stage.step}"
        )

    for stage_idx, stage in enumerate(stages):
        stage_num = stage_idx + 1
        logging.info("=" * 60)
        logging.info(f"PROGRESSIVE STAGE {stage_num}/{total_stages}")
        logging.info(f"  Patch size: {stage.patch_size}³")
        logging.info(f"  Batch size: {stage.batch_size}")
        logging.info(f"  Epochs: {stage.epochs}")
        logging.info(f"  Step: {stage.step}")
        logging.info("=" * 60)

        if settings.train.device[:4] == "cuda":
            torch.cuda.empty_cache()
            logging.info("Cleared GPU cache before stage")

        pv = _patch_volume_dim(settings)
        train_set = TrainingDataSet(
            data=noisy_data,
            patch_size=(
                pv,
                stage.patch_size,
                stage.patch_size,
                stage.patch_size,
            ),
            step=stage.step,
            mask_p=settings.train.mask_p,
            clean_data=original_data,
            brain_mask=brain_mask,
            patch_filter_method=patch_filter_method,
            min_signal_threshold=min_signal_threshold,
            **_dataset_kwargs(settings),
        )

        subset_fraction = 1
        total_samples = len(train_set)
        num_samples = int(total_samples * subset_fraction)
        np.random.seed(subset_seed)
        indices = np.random.choice(total_samples, size=num_samples, replace=False)
        train_set = Subset(train_set, indices)

        train_loader = DataLoader(train_set, batch_size=stage.batch_size, shuffle=True)
        logging.info(
            f"Stage {stage_num} DataLoader: batch_size={stage.batch_size}, "
            f"num_batches={len(train_loader)}, samples={len(train_set)}"
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=settings.train.learning_rate)
        logging.info(f"Stage {stage_num} Optimizer: Adam(lr={settings.train.learning_rate})")

        scheduler = None
        if getattr(settings.train, "use_scheduler", False):
            if settings.train.scheduler_type == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=settings.train.scheduler_step_size,
                    gamma=settings.train.scheduler_gamma,
                )
            elif settings.train.scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=min(getattr(settings.train, "scheduler_T_0", 20), stage.epochs),
                    T_mult=getattr(settings.train, "scheduler_T_mult", 2),
                    eta_min=getattr(settings.train, "eta_min_lr", 0.0001),
                )
            elif settings.train.scheduler_type == "reduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=settings.train.scheduler_patience,
                    factor=settings.train.scheduler_factor,
                    min_lr=settings.train.min_lr,
                )

        stage_checkpoint_dir = os.path.join(
            checkpoint_dir, f"stage_{stage_num}_patch{stage.patch_size}"
        )
        os.makedirs(stage_checkpoint_dir, exist_ok=True)
        stage_loss_dir = os.path.join(loss_dir, f"stage_{stage_num}_patch{stage.patch_size}")
        os.makedirs(stage_loss_dir, exist_ok=True)

        fit_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            num_epochs=stage.epochs,
            device=settings.train.device,
            checkpoint_dir=stage_checkpoint_dir,
            loss_dir=stage_loss_dir,
            use_amp=use_amp,
            supervised_mode=bool(getattr(settings.train, "supervised", False)),
        )

        logging.info(f"Stage {stage_num}/{total_stages} completed")

        if wandb.run is not None:
            wandb.log(
                {
                    "progressive/stage": stage_num,
                    "progressive/patch_size": stage.patch_size,
                    "progressive/batch_size": stage.batch_size,
                }
            )

        del train_loader, train_set
        gc.collect()
        if settings.train.device[:4] == "cuda":
            torch.cuda.empty_cache()

        if stage_num == total_stages:
            stage_best = os.path.join(stage_checkpoint_dir, "best_loss_checkpoint.pth")
            final_best = os.path.join(checkpoint_dir, "best_loss_checkpoint.pth")
            if os.path.exists(stage_best):
                shutil.copy(stage_best, final_best)
                logging.info(f"Copied final stage best checkpoint to: {final_best}")

    logging.info("Progressive learning completed successfully")


def main(
    dataset: str,
    train: bool = True,
    reconstruct: bool = True,
    generate_images: bool = True,
    noise_sigma=None,
    noise_type=None,
    noise_n_coils=None,
    config_path: str = None,
    overrides=None,
    output_root: str = None,
    registry_path: str = None,
    exp_id: str = None,
    job_id: str = None,
    recipe: str = None,
    regime: str = "self_supervised",
):
    log_file = setup_logging(log_level=logging.INFO)
    logging.info(f"Starting Restormer3D-hybrid-RGS with dataset: {dataset}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = config_path or os.path.join(script_dir, "config.yaml")

    logging.info(f"Loading config from: {config_path}")

    settings = load_config(config_path)
    logging.info("Configuration loaded successfully")
    if overrides:
        apply_overrides(settings, overrides)

    if dataset == "dbrain":
        logging.info("Using DBrain dataset configuration")
        settings = settings.dbrain
        if noise_sigma is not None:
            settings.data.noise_sigma = noise_sigma
        if noise_type is not None:
            settings.data.noise_type = noise_type
        if noise_n_coils is not None:
            settings.data.noise_n_coils = noise_n_coils
        data_loader = DBrainDataLoader(
            nii_path=settings.data.nii_path,
            bvecs_path=settings.data.bvecs_path,
            bvalue=settings.data.bvalue,
            noise_sigma=settings.data.noise_sigma,
            noise_type=getattr(settings.data, "noise_type", "rician"),
            n_coils=getattr(settings.data, "noise_n_coils", 1),
        )
        logging.info(
            f"DBrainDataLoader initialized with noise_sigma={settings.data.noise_sigma}, noise_type={getattr(settings.data, 'noise_type', 'rician')}"
        )
    elif dataset == "stanford":
        logging.info("Using Stanford dataset configuration")
        settings = settings.stanford
        data_loader = StanfordDataLoader(
            bvalue=settings.data.bvalue,
            noise_sigma=settings.data.noise_sigma,
        )
        logging.info("StanfordDataLoader initialized")
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    apply_output_root(settings, output_root)
    started_at = now_utc_iso()
    wall_t0 = time.time()
    train_secs = None
    infer_secs = None
    sec_per_epoch = None
    sec_per_volume = None
    metrics = None
    metrics_roi = None
    n_params = None
    run_status = "success"
    run_error = None

    logging.info("Setting up wandb...")
    wandb_run = None
    try:
        wandb_config = {
            "dataset": dataset,
            "model_name": "Restormer3D-hybrid-RGS",
            **settings.toDict(),
        }
        wandb_kwargs = {"project": "DWMRI-Denoising", "config": wandb_config}
        vol_seg_wandb = _volume_path_segment(settings)
        if dataset == "stanford":
            wandb_kwargs["name"] = f"{dataset}_{vol_seg_wandb}"
            wandb_kwargs["tags"] = [dataset, vol_seg_wandb]
        else:
            nt = getattr(settings.data, "noise_type", "rician")
            sigma = getattr(settings.data, "noise_sigma", 0.1)
            if sigma is not None:
                alias = (
                    "ncchi"
                    if (nt or "rician").lower().strip() == "noncentral_chi"
                    else (nt or "rician").lower().strip()
                )
                wandb_kwargs["name"] = f"{dataset}_{vol_seg_wandb}_noise_{alias}_sigma_{sigma}"
                wandb_kwargs["tags"] = [dataset, vol_seg_wandb, f"sigma_{sigma}", f"type_{alias}"]
            else:
                wandb_kwargs["name"] = f"{dataset}_{vol_seg_wandb}"
                wandb_kwargs["tags"] = [dataset, vol_seg_wandb]
        wandb_run = wandb.init(**wandb_kwargs)
        logging.info("Loading data...")
        original_data, noisy_data = data_loader.load_data()
        if original_data is None:
            original_data = noisy_data
            logging.info(
                "StanfordDataLoader returned original_data=None; using noisy_data as reference (self-supervised)"
            )
        take_volumes = _take_volumes_dwi(settings)
        logging.info(f"Taking volumes from {settings.data.num_b0s} to {take_volumes}")
        if _is_rgs(settings):
            k = int(getattr(settings.data, "num_input_volumes", settings.model.in_channel))
            if k != settings.model.in_channel:
                raise ValueError(
                    f"RGS: num_input_volumes ({k}) must match model.in_channel ({settings.model.in_channel})"
                )
        noisy_data = noisy_data[
            : settings.data.take_x,
            : settings.data.take_y,
            : settings.data.take_z,
            settings.data.num_b0s : take_volumes,
        ]
        original_data = original_data[
            : settings.data.take_x,
            : settings.data.take_y,
            : settings.data.take_z,
            settings.data.num_b0s : take_volumes,
        ]
        logging.info(f"Noisy data shape: {noisy_data.shape}")
        logging.info(
            f"Data type: {noisy_data.dtype}, Min: {noisy_data.min():.4f}, Max: {noisy_data.max():.4f}, Mean: {noisy_data.mean():.4f}"
        )

        patch_filter_method = getattr(settings.data, "patch_filter_method", "none")
        min_signal_threshold = getattr(settings.data, "min_signal_threshold", 0.0)
        otsu_median_radius = getattr(settings.data, "otsu_median_radius", 2)
        otsu_numpass = getattr(settings.data, "otsu_numpass", 1)

        logging.info(
            f"Patch filtering: method={patch_filter_method}, "
            f"threshold={min_signal_threshold}, otsu_radius={otsu_median_radius}, otsu_numpass={otsu_numpass}"
        )

        brain_mask = None
        if patch_filter_method == "otsu":
            logging.info("Computing brain mask using median_otsu...")
            brain_mask = compute_brain_mask(
                original_data,
                median_radius=otsu_median_radius,
                numpass=otsu_numpass,
            )

        logging.info("Initializing Restormer3D model...")
        model = Restormer3D(
            inp_channels=settings.model.in_channel,
            out_channels=settings.model.out_channel,
            dim=getattr(settings.model, "dim", 32),
            num_blocks=getattr(settings.model, "num_blocks", [2, 2, 2, 4]),
            num_refinement_blocks=getattr(settings.model, "num_refinement_blocks", 2),
            heads=getattr(settings.model, "heads", [1, 2, 4, 8]),
            ffn_expansion_factor=getattr(settings.model, "ffn_expansion_factor", 2.0),
            bias=getattr(settings.model, "bias", False),
            LayerNorm_type=getattr(settings.model, "LayerNorm_type", "WithBias"),
            output_activation=getattr(settings.model, "output_activation", "prelu"),
            scale_and_shift=getattr(settings.model, "scale_and_shift", True),
        )
        logging.info(
            f"Model initialized - in_channel: {settings.model.in_channel}, out_channel: {settings.model.out_channel}"
        )
        n_params = int(sum(p.numel() for p in model.parameters()))
        logging.info(f"Total model parameters: {n_params:,}")

        progressive_enabled = hasattr(settings.train, "progressive") and getattr(
            settings.train.progressive, "enabled", False
        )
        batch_for_multi_gpu = (
            settings.train.progressive.stages[0].batch_size
            if progressive_enabled
            else settings.train.batch_size
        )

        multi_gpu_config = create_multi_gpu_config_from_dict(
            {
                "multi_gpu": settings.train.multi_gpu,
                "gpu_ids": settings.train.gpu_ids,
                "auto_scale_lr": settings.train.auto_scale_lr,
                "learning_rate": settings.train.learning_rate,
                "batch_size": batch_for_multi_gpu,
                "auto_exclude_imbalanced": settings.train.auto_exclude_imbalanced,
                "memory_threshold": settings.train.memory_threshold,
            }
        )

        model, effective_lr, effective_batch_size = setup_multi_gpu(model, multi_gpu_config)

        logging.info("Setting up optimizer and scheduler...")
        optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
        logging.info(f"Optimizer: Adam(lr={effective_lr:.6f})")
        logging.info(
            f"Effective batch size: {effective_batch_size} (per-GPU: {batch_for_multi_gpu})"
        )

        scheduler = None
        if settings.train.use_scheduler:
            if settings.train.scheduler_type == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=settings.train.scheduler_step_size,
                    gamma=settings.train.scheduler_gamma,
                )
                logging.info(
                    f"Scheduler: StepLR(step_size={settings.train.scheduler_step_size}, "
                    f"gamma={settings.train.scheduler_gamma})"
                )
            elif settings.train.scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=settings.train.scheduler_T_0,
                    T_mult=settings.train.scheduler_T_mult,
                    eta_min=settings.train.eta_min_lr,
                )
                logging.info(
                    f"Scheduler: CosineAnnealingWarmRestarts(T_0={settings.train.scheduler_T_0}, T_mult={settings.train.scheduler_T_mult}, eta_min={settings.train.eta_min_lr})"
                )
            elif settings.train.scheduler_type == "reduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=settings.train.scheduler_patience,
                    factor=settings.train.scheduler_factor,
                    min_lr=settings.train.min_lr,
                )
                logging.info(
                    f"Scheduler: ReduceLROnPlateau(patience={settings.train.scheduler_patience}, factor={settings.train.scheduler_factor}, min_lr={settings.train.min_lr})"
                )

        logging.info(f"Training device: {settings.train.device}")
        logging.info(f"Number of epochs: {settings.train.num_epochs}")

        noise_segment = noise_path_segment(
            getattr(settings.data, "noise_type", "rician"),
            getattr(settings.data, "noise_sigma", 0.1),
        )
        bvalue_segment = f"b{getattr(settings.data, 'bvalue', 2500)}"
        vol_seg = _volume_path_segment(settings)
        checkpoint_dir = os.path.join(
            settings.train.checkpoint_dir,
            bvalue_segment,
            vol_seg,
            noise_segment,
            f"learning_rate_{settings.train.learning_rate}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        loss_dir = os.path.join(
            "restormer_hybrid_rgs/losses",
            dataset,
            bvalue_segment,
            vol_seg,
            noise_segment,
            f"learning_rate_{settings.train.learning_rate}",
        )
        os.makedirs(loss_dir, exist_ok=True)

        if train:
            train_t0 = time.time()
            use_amp = getattr(settings.train, "use_amp", True)

            if progressive_enabled:
                logging.info("Using progressive learning training strategy")
                fit_progressive(
                    model=model,
                    settings=settings,
                    noisy_data=noisy_data,
                    original_data=original_data,
                    brain_mask=brain_mask,
                    checkpoint_dir=checkpoint_dir,
                    loss_dir=loss_dir,
                    patch_filter_method=patch_filter_method,
                    min_signal_threshold=min_signal_threshold,
                )
            else:
                logging.info("Using standard training (progressive learning disabled)")
                pv = _patch_volume_dim(settings)
                train_set = TrainingDataSet(
                    data=noisy_data,
                    patch_size=(
                        pv,
                        settings.data.patch_size,
                        settings.data.patch_size,
                        settings.data.patch_size,
                    ),
                    step=settings.data.step,
                    mask_p=settings.train.mask_p,
                    clean_data=original_data,
                    brain_mask=brain_mask,
                    patch_filter_method=patch_filter_method,
                    min_signal_threshold=min_signal_threshold,
                    **_dataset_kwargs(settings),
                )
                subset_fraction = 0.6
                total_samples = len(train_set)
                num_samples = int(total_samples * subset_fraction)
                indices = np.random.choice(total_samples, size=num_samples, replace=False)
                train_set = Subset(train_set, indices)
                train_loader = DataLoader(
                    train_set,
                    batch_size=settings.train.batch_size,
                    shuffle=True,
                )
                logging.info(
                    f"DataLoader created with batch_size={settings.train.batch_size}, num_batches={len(train_loader)}"
                )
                fit_model(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loader=train_loader,
                    num_epochs=settings.train.num_epochs,
                    device=settings.train.device,
                    checkpoint_dir=checkpoint_dir,
                    loss_dir=loss_dir,
                    use_amp=use_amp,
                    supervised_mode=bool(getattr(settings.train, "supervised", False)),
                )

            logging.info("Training setup completed successfully")
            logging.info(f"Training completed. Log file: {log_file}")
            train_secs = time.time() - train_t0
            sec_per_epoch = float(train_secs) / float(settings.train.num_epochs)

            del model

        if reconstruct:
            infer_t0 = time.time()
            logging.info("Reconstructing DWIs...")
            best_loss_checkpoint = os.path.join(checkpoint_dir, "best_loss_checkpoint.pth")
            reconstruct_model = Restormer3D(
                inp_channels=settings.model.in_channel,
                out_channels=settings.model.out_channel,
                dim=getattr(settings.model, "dim", 32),
                num_blocks=getattr(settings.model, "num_blocks", [2, 2, 2, 4]),
                num_refinement_blocks=getattr(settings.model, "num_refinement_blocks", 2),
                heads=getattr(settings.model, "heads", [1, 2, 4, 8]),
                ffn_expansion_factor=getattr(settings.model, "ffn_expansion_factor", 2.0),
                bias=getattr(settings.model, "bias", False),
                LayerNorm_type=getattr(settings.model, "LayerNorm_type", "WithBias"),
                output_activation=getattr(settings.model, "output_activation", "prelu"),
                scale_and_shift=getattr(settings.model, "scale_and_shift", True),
            )
            reconstruct_model, _, _, _, _, _ = load_checkpoint(
                model=reconstruct_model,
                optimizer=None,
                filename=best_loss_checkpoint,
                device=settings.reconstruct.device,
                strict=False,
            )
            if n_params is None:
                n_params = int(sum(p.numel() for p in reconstruct_model.parameters()))
            x_reconstruct = torch.from_numpy(np.transpose(noisy_data, (3, 0, 1, 2))).type(
                torch.float
            )

            rec_use_amp = getattr(settings.reconstruct, "use_amp", True)
            patch_sz = getattr(settings.reconstruct, "patch_size", 32)
            overlap = getattr(settings.reconstruct, "overlap", 8)
            pred_chunk = getattr(settings.reconstruct, "pred_chunk_size", None)

            if _is_rgs(settings):
                n_ctx = int(getattr(settings.reconstruct, "n_context_samples", 10))
                reconstructed_dwis = reconstruct_dwis_rgs(
                    model=reconstruct_model,
                    data=x_reconstruct,
                    device=settings.reconstruct.device,
                    mask_p=settings.reconstruct.mask_p,
                    n_preds=settings.reconstruct.n_preds,
                    n_context=n_ctx,
                    target_channel=int(getattr(settings.data, "target_channel", 9)),
                    num_input=int(
                        getattr(
                            settings.data,
                            "num_input_volumes",
                            settings.model.in_channel,
                        )
                    ),
                    patch_size=patch_sz,
                    overlap=overlap,
                    use_amp=rec_use_amp,
                    pred_chunk_size=pred_chunk,
                )
            elif _is_sequential(settings):
                reconstructed_dwis = reconstruct_dwis_sequential_sliding_k(
                    model=reconstruct_model,
                    data=x_reconstruct,
                    device=settings.reconstruct.device,
                    mask_p=settings.reconstruct.mask_p,
                    n_preds=settings.reconstruct.n_preds,
                    target_channel=int(getattr(settings.data, "target_channel", 9)),
                    num_input=int(
                        getattr(
                            settings.data,
                            "num_input_volumes",
                            settings.model.in_channel,
                        )
                    ),
                    patch_size=patch_sz,
                    overlap=overlap,
                    use_amp=rec_use_amp,
                    pred_chunk_size=pred_chunk,
                )
            else:
                reconstructed_dwis = reconstruct_dwis(
                    model=reconstruct_model,
                    data=x_reconstruct,
                    device=settings.reconstruct.device,
                    mask_p=settings.reconstruct.mask_p,
                    n_preds=settings.reconstruct.n_preds,
                    patch_size=patch_sz,
                    overlap=overlap,
                    use_amp=rec_use_amp,
                    pred_chunk_size=pred_chunk,
                )
            reconstructed_dwis = np.transpose(reconstructed_dwis, (1, 2, 3, 0))
            logging.info(f"Reconstructed DWIs shape: {reconstructed_dwis.shape}")
            logging.info(
                f"Reconstructed DWIs min: {reconstructed_dwis.min():.4f}, "
                f"max: {reconstructed_dwis.max():.4f}, "
                f"mean: {reconstructed_dwis.mean():.4f}"
            )
            logging.info(f"Reconstructed DWIs dtype: {reconstructed_dwis.dtype}")

            if getattr(settings.reconstruct, "rescale_to_01", False):
                rescale_mode = getattr(settings.reconstruct, "rescale_mode", "per_volume")
                reference = original_data if rescale_mode == "match_gt" else None
                reconstructed_dwis = rescale_reconstruction_to_01(
                    reconstructed_dwis,
                    mode=rescale_mode,
                    reference=reference,
                )

            if getattr(settings.reconstruct, "subtract_background_estimate", False):
                thresh = getattr(settings.reconstruct, "subtract_background_threshold", 0.02)
                bg_mask = (original_data <= thresh).all(axis=-1)
                if np.any(bg_mask):
                    bg_vals = reconstructed_dwis[bg_mask]
                    shift = float(np.median(bg_vals))
                    logging.info(
                        f"Background subtraction: shift={shift:.6f} from {np.sum(bg_mask):,} voxels"
                    )
                    reconstructed_dwis = reconstructed_dwis.astype(np.float64) - shift
                    reconstructed_dwis = np.clip(reconstructed_dwis, 0, 1)

            if getattr(settings.reconstruct, "clip_to_range", False):
                reconstructed_dwis = np.clip(reconstructed_dwis, 0, 1)
                logging.info(
                    f"Clipped to [0, 1]: min={reconstructed_dwis.min():.4f}, "
                    f"max={reconstructed_dwis.max():.4f}, mean={reconstructed_dwis.mean():.4f}"
                )

            metrics_dir = os.path.join(
                settings.reconstruct.metrics_dir,
                bvalue_segment,
                vol_seg,
                noise_segment,
                f"learning_rate_{settings.train.learning_rate}",
            )
            os.makedirs(metrics_dir, exist_ok=True)

            metrics = compute_metrics(
                original_data,
                reconstructed_dwis,
            )
            logging.info(f"Metrics (full image): {metrics}")
            save_metrics(metrics, metrics_dir)

            if wandb_run is not None:
                wandb.log(
                    {
                        "reconstruct/metrics_mse": metrics["mse"],
                        "reconstruct/metrics_ssim": metrics["ssim"],
                        "reconstruct/metrics_psnr": metrics["psnr"],
                    }
                )

            roi_threshold = getattr(settings.reconstruct, "metrics_roi_threshold", None)
            if roi_threshold is not None:
                roi_mask = (original_data > roi_threshold).any(axis=-1)
                n_roi = int(np.sum(roi_mask))
                logging.info(
                    f"ROI mask: original > {roi_threshold}, {n_roi:,} voxels ({100.0 * n_roi / roi_mask.size:.1f}%)"
                )
                metrics_roi = compute_metrics(original_data, reconstructed_dwis, mask=roi_mask)
                logging.info(f"Metrics (ROI, brain/tissue only): {metrics_roi}")
                save_metrics(metrics_roi, metrics_dir, filename="metrics_roi.json")

                if wandb_run is not None:
                    wandb.log(
                        {
                            "reconstruct/metrics_roi_mse": metrics_roi["mse"],
                            "reconstruct/metrics_roi_ssim": metrics_roi["ssim"],
                            "reconstruct/metrics_roi_psnr": metrics_roi["psnr"],
                        }
                    )
            infer_secs = time.time() - infer_t0
            sec_per_volume = float(infer_secs) / float(reconstructed_dwis.shape[-1])

            if generate_images:
                logging.info("Generating images...")
                images_dir = os.path.join(
                    settings.reconstruct.images_dir,
                    bvalue_segment,
                    vol_seg,
                    noise_segment,
                    f"learning_rate_{settings.train.learning_rate}",
                )
                os.makedirs(images_dir, exist_ok=True)
                logging.info(f"Saving images to: {images_dir}")

                wandb_images = []
                n_viz = (
                    int(
                        getattr(
                            settings.data,
                            "shell_gradient_volumes",
                            settings.data.num_volumes,
                        )
                    )
                    if _is_rgs(settings)
                    else settings.data.num_volumes
                )
                for i in range(n_viz):
                    comparison_path = os.path.join(images_dir, f"comparison_volume_{i}.png")
                    fully_compare_volumes(
                        original_volume=np.transpose(original_data, (2, 3, 0, 1)),
                        noisy_volume=np.transpose(noisy_data, (2, 3, 0, 1)),
                        denoised_volume=np.transpose(reconstructed_dwis, (2, 3, 0, 1)),
                        file_name=comparison_path,
                        volume_idx=i,
                    )
                    wandb_images.append(wandb.Image(comparison_path, caption=f"Volume index {i}"))
                if wandb_run is not None:
                    wandb.log(
                        {
                            "reconstruct/comparison": wandb_images,
                        }
                    )

                single_path = os.path.join(images_dir, "single.png")
                visualize_single_volume(
                    np.transpose(reconstructed_dwis, (2, 3, 0, 1)),
                    file_name=single_path,
                    volume_idx=0,
                )

                noisy_path = os.path.join(images_dir, "noisy.png")
                visualize_single_volume(
                    np.transpose(noisy_data, (2, 3, 0, 1)),
                    file_name=noisy_path,
                    volume_idx=0,
                )

    except Exception as exc:
        run_status = "failed"
        run_error = str(exc)
        raise
    finally:
        payload = {
            "schema_version": "v1",
            "exp_id": exp_id,
            "job_id": job_id,
            "recipe": recipe,
            "status": run_status,
            "error": run_error,
            "timestamps": {"start_utc": started_at, "end_utc": now_utc_iso()},
            "duration_s": time.time() - wall_t0,
            "stage": "train_reconstruct" if (train and reconstruct) else ("train" if train else "reconstruct"),
            "dataset": dataset,
            "regime": regime,
            "architecture": "restormer",
            "dimensionality": "3d",
            "sampling_mode": getattr(settings.data, "shell_sampling_mode", "sequential"),
            "sampling_config": {
                "g_shell": int(getattr(settings.data, "shell_gradient_volumes", settings.data.num_volumes)),
                "k_input": int(getattr(settings.data, "num_input_volumes", settings.model.in_channel)),
                "target_channel": int(getattr(settings.data, "target_channel", 9)),
                "window_policy": "sliding_last_target",
            },
            "inference_config": {
                "n_context_samples": int(getattr(settings.reconstruct, "n_context_samples", 0)),
                "n_preds": int(getattr(settings.reconstruct, "n_preds", 0)),
            },
            "train_config": {
                "epochs": int(getattr(settings.train, "num_epochs", 0)),
                "batch_size": int(getattr(settings.train, "batch_size", 0)),
                "lr": float(getattr(settings.train, "learning_rate", 0.0)),
                "progressive_enabled": bool(getattr(getattr(settings.train, "progressive", {}), "enabled", False)),
            },
            "control_metrics": {
                "n_params": int(n_params or 0),
                "sec_per_epoch": sec_per_epoch,
                "sec_per_volume": sec_per_volume,
                "peak_gpu_mem_mb": gpu_peak_mem_mb(settings.reconstruct.device if reconstruct else settings.train.device),
            },
            "quality_metrics_full": metrics,
            "quality_metrics_roi": metrics_roi,
            "dti_metrics": None,
            "hardware": hardware_info(settings.reconstruct.device if reconstruct else settings.train.device),
        }
        append_registry_line(registry_path, payload)
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Restormer hybrid experiments")
    parser.add_argument("--dataset", default="dbrain", choices=["dbrain", "stanford"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--registry-path", default=None)
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--recipe", default=None)
    parser.add_argument("--regime", default="self_supervised")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-reconstruct", action="store_true")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        train=not args.skip_train,
        reconstruct=not args.skip_reconstruct,
        generate_images=not args.no_images,
        config_path=args.config,
        overrides=args.overrides,
        output_root=args.output_root,
        registry_path=args.registry_path,
        exp_id=args.exp_id,
        job_id=args.job_id,
        recipe=args.recipe,
        regime=args.regime,
    )
