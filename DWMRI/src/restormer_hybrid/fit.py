import logging
import os

import numpy as np
import torch

# L1Loss removed - using masked MSE loss for J-invariant training
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler
from tqdm import tqdm
import wandb
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.training_tracker import TrainingLossTracker


def fit_model(
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingWarmRestarts | LRScheduler | None,
    train_loader,
    num_epochs=10,
    device="cuda",
    checkpoint_dir=".",
    loss_dir=None,
):
    # #region agent log
    _log_path = "/Users/study/Documents/Repo/TechJourney/DWMRI/.cursor/debug-da8345.log"
    import json, time
    # H12: Re-enable cuDNN (more memory efficient) with benchmark for optimal kernels
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # Clear GPU cache before training
    if device == "cuda":
        torch.cuda.empty_cache()
    with open(_log_path, "a") as _f: _f.write(json.dumps({"sessionId": "da8345", "hypothesisId": "H12", "location": "fit.py:fit_model", "message": "cuDNN re-enabled, cache cleared", "data": {"cudnn_enabled": torch.backends.cudnn.enabled, "cudnn_benchmark": torch.backends.cudnn.benchmark}, "timestamp": int(time.time()*1000)}) + "\n")
    # #endregion
    logging.info((f"Starting training - device: {device}, " f"epochs: {num_epochs}"))
    logging.info(f"Model device: {next(model.parameters()).device}")

    model.to(device)
    logging.info(f"Model moved to device: {device}")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        logging.info(f"Created checkpoint directory: {checkpoint_dir}")
    else:
        logging.info(f"Using existing checkpoint directory: {checkpoint_dir}")

    # Load the latest checkpoint if it exists
    latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    model, optimizer, start_epoch, scheduler_state_dict, best_loss = load_checkpoint(
        model, optimizer, latest_checkpoint, device, strict=False
    )

    # Restore scheduler state if it exists
    if scheduler is not None and scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)
        logging.info("Scheduler state restored from checkpoint")

    logging.info(f"Training starting from epoch: {start_epoch}")
    logging.info(f"Best loss so far: {best_loss:.6f}")
    # Note: Using masked MSE loss instead of L1Loss for J-invariant training

    # Initialize loss tracker if loss_dir is provided
    loss_tracker = None
    if loss_dir is not None:
        loss_tracker = TrainingLossTracker(loss_dir)
        # Sync best loss from checkpoint if available
        tracker_best_loss, tracker_best_epoch = loss_tracker.get_best_loss()
        if tracker_best_loss < best_loss:
            best_loss = tracker_best_loss
            logging.info(
                f"Updated best loss from tracker: {best_loss:.6f} at epoch {tracker_best_epoch}"
            )

    for epoch in tqdm(
        range(start_epoch, num_epochs), desc="Training DRCnet-hybrid", total=num_epochs
    ):
        model.train()
        total_loss = 0
        batch_count = 0
        epoch_losses = []

        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Current learning rate: {current_lr:.6f}")

        for batch_idx, (x, mask, noisy_target_volume) in enumerate(train_loader):
            # x: training data is the noisy data containing all volumes
            # with a single masked volume for the target volume
            x = x.to(device)
            mask = mask.to(device)
            # noisy_target_volume: the original noisy target volume
            noisy_target_volume = noisy_target_volume.to(device)

            # Log batch information occasionally
            if batch_idx % 10 == 0:
                logging.debug(
                    (
                        f"Batch {batch_idx}/{len(train_loader)} - "
                        f"input shape: {x.shape}, mask shape: {mask.shape}, noisy_target_volume shape: {noisy_target_volume.shape}"
                    )
                )

            # forward pass
            x_recon = model(x)
            # loss: compute only on masked pixels (J-invariant loss)
            loss = torch.sum(
                (x_recon - noisy_target_volume)
                * (x_recon - noisy_target_volume)
                * (1 - mask)
            ) / torch.sum(1 - mask)
            # zero grad
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # step
            optimizer.step()

            total_loss += loss.item()
            epoch_losses.append(loss.item())
            batch_count += 1

        avg_loss = total_loss / len(train_loader)
        min_loss = min(epoch_losses)
        max_loss = max(epoch_losses)
        std_loss = np.std(epoch_losses)

        if scheduler is not None:
            scheduler.step(epoch)
        new_lr = optimizer.param_groups[0]["lr"]

        logging.info(f"Epoch {epoch+1}/{num_epochs} completed")
        logging.info(f"Average Loss: {avg_loss:.6f}")
        logging.info(
            (
                f"Loss stats - Min: {min_loss:.6f}, "
                f"Max: {max_loss:.6f}, Std: {std_loss:.6f}"
            )
        )
        logging.info(f"Learning rate: {current_lr:.6f} -> {new_lr:.6f}")

        # Record epoch in loss tracker
        if loss_tracker is not None:
            loss_tracker.record_epoch(
                epoch=epoch + 1,
                avg_loss=avg_loss,
                min_loss=min_loss,
                max_loss=max_loss,
                std_loss=std_loss,
                learning_rate=new_lr,
            )

        # Get scheduler state dict for checkpoint saving
        scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None

        # Update best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            logging.info(f"New best loss: {best_loss:.6f}")
            logging.info(f"Saving checkpoint with best loss: {best_loss:.6f}")
            best_loss_checkpoint = os.path.join(
                checkpoint_dir, "best_loss_checkpoint.pth"
            )
            save_checkpoint(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch + 1,
                loss=avg_loss,
                best_loss=best_loss,
                filename=best_loss_checkpoint,
                scheduler_state_dict=scheduler_state_dict,
            )

        # Save the latest checkpoint
        save_checkpoint(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch + 1,
            loss=avg_loss,
            best_loss=best_loss,
            filename=latest_checkpoint,
            scheduler_state_dict=scheduler_state_dict,
        )

        # Log metrics to wandb
        if wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/loss_min": min_loss,
                    "train/loss_max": max_loss,
                    "train/loss_std": std_loss,
                    "train/learning_rate": new_lr,
                    "train/best_loss": best_loss,
                }
            )

    logging.info("Training completed successfully.")
    logging.info(f"Final best loss: {best_loss:.6f}")
    logging.info(f"Total epochs trained: {num_epochs - start_epoch}")
