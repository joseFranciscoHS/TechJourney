import argparse
import gc
import logging
import os
import shutil
import time

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from drcnet_hybrid_rgs.data import TrainingDataSet
from drcnet_hybrid_rgs.fit import fit_model
from drcnet_hybrid_rgs.model import DenoiserNet
from drcnet_hybrid_rgs.reconstruction import (
    reconstruct_dwis_rgs,
    reconstruct_dwis_sequential_sliding_k,
)
from paper_eval.dti_metrics import save_dti_metrics, try_compute_dti_errors
from paper_eval.export_denoised import maybe_export_denoised
from utils import setup_logging
from utils.checkpoint import load_checkpoint
from utils.data import (
    DBrainDataLoader,
    StanfordDataLoader,
    compute_brain_mask,
    invert_normalization,
)
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
from utils.metrics import (
    compute_metrics,
    fully_compare_volumes,
    save_metrics,
    visualize_single_volume,
)
from utils.multi_gpu import create_multi_gpu_config_from_dict, setup_multi_gpu
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

_OBJECTIVE_MODES = {"hybrid", "angular", "spatial"}


def _is_rgs(settings) -> bool:
    return getattr(settings.data, "shell_sampling_mode", "sequential") == "rgs"


def _is_sequential(settings) -> bool:
    return getattr(settings.data, "shell_sampling_mode", "sequential") == "sequential"


def _objective_mode(settings) -> str:
    mode = str(getattr(settings.train, "objective_mode", "hybrid")).lower()
    if mode not in _OBJECTIVE_MODES:
        raise ValueError(
            f"train.objective_mode must be one of {sorted(_OBJECTIVE_MODES)}, got {mode!r}"
        )
    return mode


def _volume_path_segment(settings) -> str:
    objective = _objective_mode(settings)
    if _is_rgs(settings):
        g = int(
            getattr(settings.data, "shell_gradient_volumes", settings.data.num_volumes)
        )
        k = int(getattr(settings.data, "num_input_volumes", settings.model.in_channel))
        segment = f"rgs_G{g}_K{k}"
        return segment if objective == "hybrid" else f"{objective}_{segment}"
    if _is_sequential(settings):
        g = int(
            getattr(settings.data, "shell_gradient_volumes", settings.data.num_volumes)
        )
        k = int(getattr(settings.data, "num_input_volumes", settings.model.in_channel))
        segment = f"sequential_G{g}_K{k}"
        return segment if objective == "hybrid" else f"{objective}_{segment}"
    segment = f"num_volumes_{settings.data.num_volumes}"
    return segment if objective == "hybrid" else f"{objective}_{segment}"


def _validate_objective_settings(settings) -> str:
    objective = _objective_mode(settings)
    if objective != "hybrid" and not _is_rgs(settings):
        raise ValueError(
            f"train.objective_mode={objective!r} is supported only with data.shell_sampling_mode='rgs'"
        )
    if objective != "hybrid" and bool(
        getattr(settings.model, "use_film_conditioning", False)
    ):
        raise ValueError(
            "Angular/spatial objective modes do not support model.use_film_conditioning=true yet"
        )

    if _is_rgs(settings) or _is_sequential(settings):
        k = int(getattr(settings.data, "num_input_volumes", settings.model.in_channel))
        if k != settings.model.in_channel:
            raise ValueError(
                f"num_input_volumes ({k}) must match model.in_channel ({settings.model.in_channel})"
            )
        g = int(
            getattr(settings.data, "shell_gradient_volumes", settings.data.num_volumes)
        )
        if objective == "angular" and k >= g:
            raise ValueError(
                f"angular objective requires K={k} to be smaller than shell size G={g}"
            )
        if objective == "spatial" and k != 1:
            raise ValueError(f"spatial objective requires K=1, got K={k}")
        target_channel = int(getattr(settings.data, "target_channel", 9))
        if objective == "hybrid" and not (0 <= target_channel < k):
            raise ValueError(f"target_channel={target_channel} must be in [0, {k - 1}]")

    return objective


def _patch_volume_dim(settings) -> int:
    if _is_rgs(settings) or _is_sequential(settings):
        return int(
            getattr(settings.data, "num_input_volumes", settings.model.in_channel)
        )
    return settings.data.num_volumes


def _take_volumes_dwi(settings) -> int:
    if _is_rgs(settings) or _is_sequential(settings):
        return settings.data.num_b0s + int(
            getattr(settings.data, "shell_gradient_volumes", settings.data.num_volumes)
        )
    return settings.data.num_b0s + settings.data.num_volumes


def _dataset_kwargs(settings):
    mode = getattr(settings.data, "shell_sampling_mode", "sequential")
    kw = {"shell_sampling_mode": mode, "objective_mode": _objective_mode(settings)}
    if mode in {"rgs", "sequential"}:
        kw["num_input_volumes"] = int(
            getattr(settings.data, "num_input_volumes", settings.model.in_channel)
        )
        kw["target_channel"] = int(getattr(settings.data, "target_channel", 9))
    return kw


def _training_sample_kwargs(settings):
    kw = _dataset_kwargs(settings)
    kw["sample_rng_seed"] = int(getattr(settings.train, "seed", 42))
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
    dl_generator,
    cudnn_fast: bool,
    bvecs=None,
    bvals=None,
):
    """
    Train model progressively with increasing patch sizes.

    Progressive learning stages use per-stage patch_size, step, batch_size, and
    epochs; each stage gets a new dataset, optimizer, and scheduler.
    """
    stages = settings.train.progressive.stages
    total_stages = len(stages)
    use_amp = getattr(settings.train, "use_amp", True)

    logging.info(
        f"Progressive Learning: {total_stages} stages, AMP={'enabled' if use_amp else 'disabled'}"
    )
    for i, stage in enumerate(stages):
        logging.info(
            f"  Stage {i + 1}: patch={stage.patch_size}³, batch={stage.batch_size}, "
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
            bvecs=bvecs,
            bvals=bvals,
            **_training_sample_kwargs(settings),
        )

        train_set, _n_tot, _n_used = apply_training_patch_subset_from_train_block(
            train_set, settings.train
        )

        train_loader = DataLoader(
            train_set,
            batch_size=stage.batch_size,
            shuffle=True,
            generator=dl_generator,
        )
        logging.info(
            f"Stage {stage_num} DataLoader: batch_size={stage.batch_size}, "
            f"num_batches={len(train_loader)}, samples={len(train_set)}"
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=settings.train.learning_rate
        )
        logging.info(
            f"Stage {stage_num} Optimizer: Adam(lr={settings.train.learning_rate})"
        )

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
        stage_loss_dir = os.path.join(
            loss_dir, f"stage_{stage_num}_patch{stage.patch_size}"
        )
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
            objective_mode=_objective_mode(settings),
            cudnn_fast=cudnn_fast,
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

        # Free memory before next stage so the next TrainingDataSet allocation can succeed
        del train_loader, train_set
        gc.collect()
        if settings.train.device[:4] == "cuda":
            torch.cuda.empty_cache()

        if stage_num == total_stages:
            stage_best = os.path.join(stage_checkpoint_dir, "best_loss_checkpoint.pth")
            final_best = os.path.join(checkpoint_dir, "best_loss_checkpoint.pth")
            stage_latest = os.path.join(stage_checkpoint_dir, "latest_checkpoint.pth")
            if os.path.exists(stage_best):
                shutil.copy(stage_best, final_best)
                logging.info(f"Copied final stage best checkpoint to: {final_best}")
            elif os.path.exists(stage_latest):
                logging.warning(
                    "Final stage has no best_loss_checkpoint.pth; copying latest_checkpoint.pth "
                    "to best_loss_checkpoint for reconstruction."
                )
                shutil.copy(stage_latest, stage_best)
                shutil.copy(stage_latest, final_best)

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
    use_wandb: bool = True,
    checkpoint_path: str = None,
):
    # Setup logging
    log_file = setup_logging(log_level=logging.INFO)
    logging.info(f"Starting training with dataset: {dataset}")

    # Get the directory where this script is located
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
        data_loader = StanfordDataLoader()
        logging.info("StanfordDataLoader initialized")
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    apply_output_root(settings, output_root)
    objective_mode = _validate_objective_settings(settings)

    train_seed = int(getattr(settings.train, "seed", 42))
    cudnn_fast = not bool(getattr(settings.train, "reproducible", False))
    set_seed(train_seed)
    configure_cudnn(fast=cudnn_fast)
    log_runtime_env(settings.train.device)
    dl_generator = make_dataloader_generator(train_seed)

    started_at = now_utc_iso()
    wall_t0 = time.time()
    train_secs = None
    infer_secs = None
    sec_per_epoch = None
    sec_per_volume = None
    n_params = None
    metrics = None
    metrics_roi = None
    dti_metrics = None
    run_status = "success"
    run_error = None
    original_xyzv_b0 = None

    logging.info("Setting up wandb...")
    wandb_run = None
    try:
        wandb_config = {
            "dataset": dataset,
            "model_name": "DRCNet-hybrid-rgs",
            **settings.toDict(),
        }
        wandb_kwargs = {"project": "DWMRI-Denoising", "config": wandb_config}
        wandb_kwargs["name"] = f"{dataset}_{_volume_path_segment(settings)}"
        wandb_kwargs["tags"] = [dataset, _volume_path_segment(settings), objective_mode]
        if use_wandb:
            wandb_run = wandb.init(**wandb_kwargs)
        else:
            logging.info("WandB disabled (use_wandb=False / --no-wandb)")
        logging.info("Loading data...")
        original_data, noisy_data = data_loader.load_data()
        # Stanford loader returns (None, data); treat as self-supervised training where GT=noisy
        if original_data is None:
            original_data = noisy_data
            logging.info(
                "StanfordDataLoader returned original_data=None; using noisy_data as reference (self-supervised)"
            )
        # omitting the b0s from the data
        take_volumes = _take_volumes_dwi(settings)
        logging.info(f"Taking volumes from {settings.data.num_b0s} to {take_volumes}")
        tx, ty, tz = settings.data.take_x, settings.data.take_y, settings.data.take_z
        original_xyzv_b0 = original_data[:tx, :ty, :tz, :take_volumes]
        noisy_data = noisy_data[:tx, :ty, :tz, settings.data.num_b0s : take_volumes]
        original_data = original_data[
            :tx, :ty, :tz, settings.data.num_b0s : take_volumes
        ]
        logging.info(f"Noisy data shape: {noisy_data.shape}")
        logging.info(
            f"Data type: {noisy_data.dtype}, Min: {noisy_data.min():.4f}, Max: {noisy_data.max():.4f}, Mean: {noisy_data.mean():.4f}"
        )

        use_film_conditioning = bool(
            getattr(settings.model, "use_film_conditioning", False)
        )
        orientation_bvecs = None
        orientation_bvals = None
        if use_film_conditioning:
            logging.info("Loading gradient table for orientation encoding")
            gtab = data_loader.load_gradient_table()
            orientation_bvecs = np.asarray(gtab.bvecs)[
                settings.data.num_b0s : take_volumes
            ]
            orientation_bvals = np.asarray(gtab.bvals)[
                settings.data.num_b0s : take_volumes
            ]
            if orientation_bvecs.shape[0] != noisy_data.shape[-1]:
                raise ValueError(
                    "Orientation metadata length does not match DWI volume count: "
                    f"{orientation_bvecs.shape[0]} vs {noisy_data.shape[-1]}"
                )

        # Patch filtering configuration
        patch_filter_method = getattr(settings.data, "patch_filter_method", "none")
        min_signal_threshold = getattr(settings.data, "min_signal_threshold", 0.0)
        otsu_median_radius = getattr(settings.data, "otsu_median_radius", 2)
        otsu_numpass = getattr(settings.data, "otsu_numpass", 1)

        logging.info(
            f"Patch filtering: method={patch_filter_method}, "
            f"threshold={min_signal_threshold}, otsu_radius={otsu_median_radius}, otsu_numpass={otsu_numpass}"
        )

        # Compute brain mask if using otsu method
        brain_mask = None
        if patch_filter_method == "otsu":
            logging.info("Computing brain mask using median_otsu...")
            brain_mask = compute_brain_mask(
                original_data,
                median_radius=otsu_median_radius,
                numpass=otsu_numpass,
            )

        progressive_enabled = hasattr(settings.train, "progressive") and getattr(
            settings.train.progressive, "enabled", False
        )
        noise_segment = noise_path_segment(
            getattr(settings.data, "noise_type", "rician"),
            getattr(settings.data, "noise_sigma", 0.1),
        )
        bvalue_segment = f"b{getattr(settings.data, 'bvalue', 2500)}"
        vol_seg = _volume_path_segment(settings)
        _sub_seg = training_subset_checkpoint_segment(settings.train)
        _path_mid = [bvalue_segment, vol_seg] + ([_sub_seg] if _sub_seg else [])
        checkpoint_dir = os.path.join(
            settings.train.checkpoint_dir,
            *_path_mid,
            noise_segment,
            f"learning_rate_{settings.train.learning_rate}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        # setting loss dir taking into account run/model parameters (includes bvalue)
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

        if train:
            train_t0 = time.time()
            use_amp = getattr(settings.train, "use_amp", True)
            if progressive_enabled:
                logging.info("Using progressive learning training strategy")
                first_stage = settings.train.progressive.stages[0]
                logging.info(
                    "Initializing DenoiserNet model (first stage patch size)..."
                )
                model = DenoiserNet(
                    input_channels=settings.model.in_channel,
                    output_channels=settings.model.out_channel,
                    groups=settings.model.groups,
                    dense_convs=settings.model.dense_convs,
                    residual=settings.model.residual,
                    base_filters=settings.model.base_filters,
                    output_shape=(
                        settings.model.out_channel,
                        first_stage.patch_size,
                        first_stage.patch_size,
                        first_stage.patch_size,
                    ),
                    device=settings.train.device,
                    output_activation=getattr(
                        settings.model, "output_activation", "prelu"
                    ),
                    use_film_conditioning=bool(
                        getattr(settings.model, "use_film_conditioning", False)
                    ),
                    film_hidden_dim=int(getattr(settings.model, "film_hidden_dim", 32)),
                    target_channel=int(getattr(settings.data, "target_channel", 15)),
                )
                logging.info(
                    f"Model initialized - in_channel: {settings.model.in_channel}, "
                    f"out_channel: {settings.model.out_channel}"
                )
                logging.info(
                    f"Total model parameters: {sum(p.numel() for p in model.parameters())}"
                )
                n_params = int(sum(p.numel() for p in model.parameters()))
                multi_gpu_config = create_multi_gpu_config_from_dict(
                    {
                        "multi_gpu": settings.train.multi_gpu,
                        "gpu_ids": settings.train.gpu_ids,
                        "auto_scale_lr": settings.train.auto_scale_lr,
                        "learning_rate": settings.train.learning_rate,
                        "batch_size": first_stage.batch_size,
                        "auto_exclude_imbalanced": settings.train.auto_exclude_imbalanced,
                        "memory_threshold": settings.train.memory_threshold,
                    }
                )
                model, _, _ = setup_multi_gpu(model, multi_gpu_config)
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
                    dl_generator=dl_generator,
                    cudnn_fast=cudnn_fast,
                    bvecs=orientation_bvecs,
                    bvals=orientation_bvals,
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
                    bvecs=orientation_bvecs,
                    bvals=orientation_bvals,
                    **_training_sample_kwargs(settings),
                )
                train_set, _n_tot, _n_used = (
                    apply_training_patch_subset_from_train_block(
                        train_set, settings.train
                    )
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=settings.train.batch_size,
                    shuffle=True,
                    generator=dl_generator,
                )
                logging.info(
                    f"DataLoader created with batch_size={settings.train.batch_size}, "
                    f"num_batches={len(train_loader)}"
                )
                logging.info("Initializing DenoiserNet model...")
                model = DenoiserNet(
                    input_channels=settings.model.in_channel,
                    output_channels=settings.model.out_channel,
                    groups=settings.model.groups,
                    dense_convs=settings.model.dense_convs,
                    residual=settings.model.residual,
                    base_filters=settings.model.base_filters,
                    output_shape=(
                        settings.model.out_channel,
                        settings.data.patch_size,
                        settings.data.patch_size,
                        settings.data.patch_size,
                    ),
                    device=settings.train.device,
                    output_activation=getattr(
                        settings.model, "output_activation", "prelu"
                    ),
                    use_film_conditioning=bool(
                        getattr(settings.model, "use_film_conditioning", False)
                    ),
                    film_hidden_dim=int(getattr(settings.model, "film_hidden_dim", 32)),
                    target_channel=int(getattr(settings.data, "target_channel", 15)),
                )
                logging.info(
                    f"Model initialized - in_channel: {settings.model.in_channel}, "
                    f"out_channel: {settings.model.out_channel}"
                )
                logging.info(
                    f"Total model parameters: {sum(p.numel() for p in model.parameters())}"
                )
                n_params = int(sum(p.numel() for p in model.parameters()))
                multi_gpu_config = create_multi_gpu_config_from_dict(
                    {
                        "multi_gpu": settings.train.multi_gpu,
                        "gpu_ids": settings.train.gpu_ids,
                        "auto_scale_lr": settings.train.auto_scale_lr,
                        "learning_rate": settings.train.learning_rate,
                        "batch_size": settings.train.batch_size,
                        "auto_exclude_imbalanced": settings.train.auto_exclude_imbalanced,
                        "memory_threshold": settings.train.memory_threshold,
                    }
                )
                model, effective_lr, effective_batch_size = setup_multi_gpu(
                    model, multi_gpu_config
                )
                logging.info("Setting up optimizer and scheduler...")
                optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
                logging.info(f"Optimizer: Adam(lr={effective_lr:.6f})")
                logging.info(
                    f"Effective batch size: {effective_batch_size} "
                    f"(per-GPU: {settings.train.batch_size})"
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
                        scheduler = (
                            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                optimizer,
                                T_0=settings.train.scheduler_T_0,
                                T_mult=settings.train.scheduler_T_mult,
                                eta_min=settings.train.eta_min_lr,
                            )
                        )
                        logging.info(
                            f"Scheduler: CosineAnnealingWarmRestarts(T_0={settings.train.scheduler_T_0}, "
                            f"T_mult={settings.train.scheduler_T_mult}, eta_min={settings.train.eta_min_lr})"
                        )
                    elif settings.train.scheduler_type == "reduceLROnPlateau":
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            patience=settings.train.scheduler_patience,
                            factor=settings.train.scheduler_factor,
                            min_lr=settings.train.min_lr,
                        )
                        logging.info(
                            f"Scheduler: ReduceLROnPlateau(patience={settings.train.scheduler_patience}, "
                            f"factor={settings.train.scheduler_factor}, min_lr={settings.train.min_lr})"
                        )
                logging.info(f"Training device: {settings.train.device}")
                logging.info(f"Number of epochs: {settings.train.num_epochs}")
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
                    objective_mode=objective_mode,
                    cudnn_fast=cudnn_fast,
                )
            train_secs = time.time() - train_t0
            sec_per_epoch = float(train_secs) / float(settings.train.num_epochs)
            logging.info("Training setup completed successfully")
            logging.info(f"Training completed. Log file: {log_file}")

            del model

        if reconstruct:
            infer_t0 = time.time()
            logging.info("Reconstructing DWIs...")
            best_loss_checkpoint = checkpoint_path or os.path.join(
                checkpoint_dir, "best_loss_checkpoint.pth"
            )
            if checkpoint_path:
                logging.info("Using checkpoint: %s", best_loss_checkpoint)
            reconstruct_model = DenoiserNet(
                input_channels=settings.model.in_channel,
                output_channels=settings.model.out_channel,
                groups=settings.model.groups,
                dense_convs=settings.model.dense_convs,
                residual=settings.model.residual,
                base_filters=settings.model.base_filters,
                output_shape=(
                    settings.model.out_channel,
                    settings.data.take_x,
                    settings.data.take_y,
                    settings.data.take_z,
                ),
                device=settings.train.device,
                output_activation=getattr(settings.model, "output_activation", "prelu"),
                use_film_conditioning=bool(
                    getattr(settings.model, "use_film_conditioning", False)
                ),
                film_hidden_dim=int(getattr(settings.model, "film_hidden_dim", 32)),
                target_channel=int(getattr(settings.data, "target_channel", 15)),
            )
            reconstruct_model, _, _, _, _, _ = load_checkpoint(
                model=reconstruct_model,
                optimizer=None,
                filename=best_loss_checkpoint,
                device=settings.reconstruct.device,
                strict=False,  # Allow partial loading for architecture changes
            )
            if n_params is None:
                n_params = int(sum(p.numel() for p in reconstruct_model.parameters()))
            # Prepare data for reconstruction: transpose from (X, Y, Z, Vols) to (Vols, X, Y, Z)
            x_reconstruct = torch.from_numpy(
                np.transpose(noisy_data, (3, 0, 1, 2))
            ).type(torch.float)

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
                    pred_chunk_size=getattr(
                        settings.reconstruct, "pred_chunk_size", None
                    ),
                    bvecs=orientation_bvecs,
                    bvals=orientation_bvals,
                    objective_mode=objective_mode,
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
                    seed=int(getattr(settings.train, "seed", 42)),
                    pred_chunk_size=getattr(
                        settings.reconstruct, "pred_chunk_size", None
                    ),
                    bvecs=orientation_bvecs,
                    bvals=orientation_bvals,
                    objective_mode=objective_mode,
                )
            else:
                raise ValueError(
                    "Sequential reconstruction not supported for Stanford few-volume dataset."
                )
            # Transpose back to (X, Y, Z, Vols) for metrics and visualization
            reconstructed_dwis = np.transpose(reconstructed_dwis, (1, 2, 3, 0))
            logging.info(f"Reconstructed DWIs shape: {reconstructed_dwis.shape}")
            logging.info(
                f"Reconstructed DWIs min: {reconstructed_dwis.min():.4f}, "
                f"max: {reconstructed_dwis.max():.4f}, "
                f"mean: {reconstructed_dwis.mean():.4f}"
            )
            logging.info(f"Reconstructed DWIs dtype: {reconstructed_dwis.dtype}")

            # Optional: subtract estimated background level then clip
            if getattr(settings.reconstruct, "subtract_background_estimate", False):
                thresh = getattr(
                    settings.reconstruct, "subtract_background_threshold", 0.02
                )
                bg_mask = (original_data <= thresh).all(axis=-1)
                if np.any(bg_mask):
                    bg_vals = reconstructed_dwis[bg_mask]
                    shift = float(np.median(bg_vals))
                    logging.info(
                        f"Background subtraction: shift={shift:.6f} from {np.sum(bg_mask):,} voxels"
                    )
                    reconstructed_dwis = reconstructed_dwis.astype(np.float64) - shift
                    reconstructed_dwis = np.clip(reconstructed_dwis, 0, 1)

            # Native (pre-eval-protocol) reconstruction, captured for the denoised
            # NIfTI/.npy export (correctness rule 2: per-volume rescale below would
            # distort each volume's S(g)/S0 and corrupt downstream CSD peaks).
            reconstructed_dwis_native = reconstructed_dwis

            rescale_to_01 = bool(getattr(settings.reconstruct, "rescale_to_01", False))
            rescale_mode = str(
                getattr(settings.reconstruct, "rescale_mode", "per_volume")
            )
            clip_to_range = bool(getattr(settings.reconstruct, "clip_to_range", False))
            reconstructed_dwis = apply_reconstruction_eval_protocol(
                reconstructed_dwis,
                original_data,
                rescale_to_01=rescale_to_01,
                rescale_mode=rescale_mode,
                clip_to_range=clip_to_range,
            )

            # setting metrics dir taking into account run/model parameters (includes bvalue)
            metrics_dir = os.path.join(
                settings.reconstruct.metrics_dir,
                bvalue_segment,
                vol_seg,
                noise_segment,
                f"learning_rate_{settings.train.learning_rate}",
            )
            os.makedirs(metrics_dir, exist_ok=True)

            # Full-image metrics
            metrics = compute_metrics(
                original_data,
                reconstructed_dwis,
            )
            logging.info(f"Metrics (full image): {metrics}")
            save_metrics(metrics, metrics_dir)

            # Log metrics to wandb
            if wandb_run is not None:
                wandb.log(
                    {
                        "reconstruct/metrics_mse": metrics["mse"],
                        "reconstruct/metrics_ssim": metrics["ssim"],
                        "reconstruct/metrics_psnr": metrics["psnr"],
                    }
                )

            # ROI-based metrics (brain/tissue only)
            roi_threshold = getattr(settings.reconstruct, "metrics_roi_threshold", None)
            roi_mask = compute_roi_mask(original_data, roi_threshold)
            if roi_mask is not None:
                n_roi, roi_pct = summarize_roi(roi_mask)
                logging.info(
                    f"ROI mask: original > {roi_threshold}, {n_roi:,} voxels ({roi_pct:.1f}%)"
                )
                metrics_roi = compute_metrics(
                    original_data, reconstructed_dwis, mask=roi_mask
                )
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

            if original_xyzv_b0 is not None and getattr(
                settings.reconstruct, "compute_dti", True
            ):
                try:
                    gtab = data_loader.load_gradient_table()
                    bvals = np.asarray(gtab.bvals)[:take_volumes]
                    bvecs = np.asarray(gtab.bvecs)[:take_volumes]
                    nb0 = int(settings.data.num_b0s)
                    gt_xyzv = original_xyzv_b0.astype(np.float64)
                    den_xyzv = np.concatenate(
                        [gt_xyzv[..., :nb0], reconstructed_dwis.astype(np.float64)],
                        axis=-1,
                    )
                    norm_params = getattr(data_loader, "norm_params_", None)
                    if norm_params is not None:
                        gt_xyzv = invert_normalization(
                            gt_xyzv, norm_params[:take_volumes]
                        )
                        den_dwis = invert_normalization(
                            reconstructed_dwis.astype(np.float64),
                            norm_params[nb0:take_volumes],
                        )
                        den_xyzv = np.concatenate(
                            [gt_xyzv[..., :nb0], den_dwis.astype(np.float64)],
                            axis=-1,
                        )
                    roi_thr = getattr(
                        settings.reconstruct, "metrics_roi_threshold", 0.02
                    )
                    dti_metrics = try_compute_dti_errors(
                        den_xyzv,
                        gt_xyzv,
                        bvals,
                        bvecs,
                        roi_threshold=roi_thr,
                    )
                    save_dti_metrics(dti_metrics, metrics_dir)
                    logging.info("DTI metrics: %s", dti_metrics)
                    if wandb_run is not None:
                        wandb.log({f"dti/{k}": v for k, v in dti_metrics.items()})
                except Exception as dti_exc:
                    logging.warning("DTI metrics skipped: %s", dti_exc)
                    dti_metrics = {
                        "fa_mae": None,
                        "md_mae": None,
                        "ad_mae": None,
                        "rd_mae": None,
                        "dti_reference": "clean_gt",
                        "dti_skipped_reason": str(dti_exc),
                    }
                    save_dti_metrics(dti_metrics, metrics_dir)
            else:
                dti_metrics = {
                    "fa_mae": None,
                    "md_mae": None,
                    "ad_mae": None,
                    "rd_mae": None,
                    "dti_reference": "clean_gt"
                    if original_xyzv_b0 is not None
                    else "self_reference_noisy",
                    "dti_skipped_reason": "no_clean_gt_or_compute_dti_false",
                }
                save_dti_metrics(dti_metrics, metrics_dir)

            # Optional denoised array export for downstream CSD fixel study
            # (no-op unless reconstruct.save_denoised_{npy,nifti} is set).
            maybe_export_denoised(
                settings,
                dataset,
                job_id or "drcnet3d",
                reconstructed_dwis_native,
                original_xyzv_b0,
                data_loader,
                take_volumes,
            )

            policy = metrics_policy_dict(
                reference_name="clean_gt"
                if dataset == "dbrain"
                else "self_reference_noisy",
                rescale_to_01=rescale_to_01,
                rescale_mode=rescale_mode,
                clip_to_range=clip_to_range,
                roi_threshold=roi_threshold,
            )
            save_run_manifest(
                out_dir=metrics_dir,
                seed=int(getattr(settings.train, "seed", 42)),
                reproducible=bool(getattr(settings.train, "reproducible", False)),
                runtime_device=str(settings.reconstruct.device),
                config={
                    "dataset": dataset,
                    "architecture": "drcnet_hybrid_rgs",
                    "objective_mode": objective_mode,
                    "sampling_mode": getattr(
                        settings.data, "shell_sampling_mode", "sequential"
                    ),
                    "k_input": int(
                        getattr(
                            settings.data,
                            "num_input_volumes",
                            settings.model.in_channel,
                        )
                    ),
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
                },
                metrics_policy=policy,
            )

            infer_secs = time.time() - infer_t0
            sec_per_volume = float(infer_secs) / float(reconstructed_dwis.shape[-1])

            if generate_images:
                logging.info("Generating images...")
                # setting images dir taking into account run/model parameters (includes bvalue)
                images_dir = os.path.join(
                    settings.reconstruct.images_dir,
                    bvalue_segment,
                    vol_seg,
                    noise_segment,
                    f"learning_rate_{settings.train.learning_rate}",
                )
                os.makedirs(images_dir, exist_ok=True)
                logging.info(f"Saving images to: {images_dir}")

                # Generate comparison image
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
                    comparison_path = os.path.join(
                        images_dir, f"comparison_volume_{i}.png"
                    )
                    fully_compare_volumes(
                        original_volume=np.transpose(original_data, (2, 3, 0, 1)),
                        noisy_volume=np.transpose(noisy_data, (2, 3, 0, 1)),
                        denoised_volume=np.transpose(reconstructed_dwis, (2, 3, 0, 1)),
                        file_name=comparison_path,
                        volume_idx=i,
                    )
                    wandb_images.append(
                        wandb.Image(comparison_path, caption=f"Volume index {i}")
                    )
                # Log images to wandb
                if wandb_run is not None:
                    wandb.log(
                        {
                            "reconstruct/comparison": wandb_images,
                        }
                    )

                # Generate single volume images
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
            "stage": "train_reconstruct"
            if (train and reconstruct)
            else ("train" if train else "reconstruct"),
            "dataset": dataset,
            "regime": regime,
            "architecture": "drcnet",
            "dimensionality": "3d",
            "objective_mode": objective_mode,
            "sampling_mode": getattr(
                settings.data, "shell_sampling_mode", "sequential"
            ),
            "sampling_config": {
                "g_shell": int(
                    getattr(
                        settings.data,
                        "shell_gradient_volumes",
                        settings.data.num_volumes,
                    )
                ),
                "k_input": int(
                    getattr(
                        settings.data, "num_input_volumes", settings.model.in_channel
                    )
                ),
                "target_channel": int(getattr(settings.data, "target_channel", 9)),
                "window_policy": "sliding_last_target"
                if objective_mode == "hybrid"
                else f"{objective_mode}_target_policy",
            },
            "objective_config": {
                "mode": objective_mode,
                "uses_angular_context": objective_mode in {"hybrid", "angular"},
                "uses_spatial_masked_target": objective_mode in {"hybrid", "spatial"},
                "loss": "full_mse"
                if bool(getattr(settings.train, "supervised", False))
                or objective_mode == "angular"
                else "masked_mse",
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
                "n_params": int(n_params or 0),
                "sec_per_epoch": sec_per_epoch,
                "sec_per_volume": sec_per_volume,
                "peak_gpu_mem_mb": gpu_peak_mem_mb(
                    settings.reconstruct.device
                    if reconstruct
                    else settings.train.device
                ),
            },
            "quality_metrics_full": metrics,
            "quality_metrics_roi": metrics_roi,
            "dti_metrics": dti_metrics,
            "hardware": hardware_info(
                settings.reconstruct.device if reconstruct else settings.train.device
            ),
        }
        append_registry_line(registry_path, payload)
        # Ensure wandb run is always finished, even if an exception occurs
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DRCNet hybrid experiments")
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
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Do not initialize WandB (useful for batch/cluster without credentials).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to best_loss_checkpoint.pth (for --skip-train / inference-only jobs).",
    )
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
        use_wandb=not args.no_wandb,
        checkpoint_path=args.checkpoint,
    )
