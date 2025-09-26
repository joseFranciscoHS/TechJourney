import logging
import os

import numpy as np
import torch
from torch.nn import L1Loss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler
from tqdm import tqdm
from utils.checkpoint import load_checkpoint, save_checkpoint
from drcnet_sinusoidal_volume_encoder.model import EdgeAwareLoss


def fit_model(
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingWarmRestarts | LRScheduler | None,
    train_loader,
    num_epochs=10,
    device="cuda",
    checkpoint_dir=".",
    use_edge_aware_loss=True,
    edge_loss_alpha=0.5,
):
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

    # Initialize loss function
    if use_edge_aware_loss:
        loss_fn = EdgeAwareLoss(alpha=edge_loss_alpha)
        logging.info(f"Using EdgeAwareLoss with alpha={edge_loss_alpha}")
    else:
        loss_fn = L1Loss()
        logging.info("Using L1Loss")

    # Move loss function to device
    loss_fn.to(device)

    for epoch in tqdm(
        range(start_epoch, num_epochs), desc="Training DRCnet", total=num_epochs
    ):
        model.train()
        total_loss = 0
        batch_count = 0
        epoch_losses = []

        # Track loss components for EdgeAwareLoss
        if use_edge_aware_loss:
            total_mse_loss = 0
            total_edge_loss = 0

        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Current learning rate: {current_lr:.6f}")

        for batch_idx, (x, y, volume_indices) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            volume_indices = volume_indices.to(device)

            # Log batch information occasionally
            if batch_idx % 10 == 0:
                logging.debug(
                    (
                        f"Batch {batch_idx}/{len(train_loader)} - "
                        f"input shape: {x.shape}, volume_indices shape: {volume_indices.shape}"
                    )
                )

            # forward pass with volume indices
            x_recon = model(x, volume_indices)
            # loss
            loss = loss_fn(x_recon, y)

            # Track loss components for EdgeAwareLoss
            if use_edge_aware_loss:
                # Extract individual loss components for logging
                with torch.no_grad():
                    mse_loss = torch.nn.functional.mse_loss(x_recon, y)
                    pred_edges = loss_fn.detect_edges(x_recon)
                    target_edges = loss_fn.detect_edges(y)
                    edge_loss = torch.nn.functional.mse_loss(pred_edges, target_edges)

                    total_mse_loss += mse_loss.item()
                    total_edge_loss += edge_loss.item()

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

        # Log detailed loss components for EdgeAwareLoss
        if use_edge_aware_loss:
            avg_mse_loss = total_mse_loss / len(train_loader)
            avg_edge_loss = total_edge_loss / len(train_loader)
            logging.info(
                f"Loss Components - MSE: {avg_mse_loss:.6f}, Edge: {avg_edge_loss:.6f}"
            )
            logging.info(f"Edge loss ratio: {avg_edge_loss/avg_mse_loss:.4f}")

        logging.info(
            (
                f"Loss stats - Min: {min_loss:.6f}, "
                f"Max: {max_loss:.6f}, Std: {std_loss:.6f}"
            )
        )
        logging.info(f"Learning rate: {current_lr:.6f} -> {new_lr:.6f}")

        # Update best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            logging.info(f"New best loss: {best_loss:.6f}")
            logging.info(f"Saving checkpoint with best loss: {best_loss:.6f}")
            best_loss_checkpoint = os.path.join(
                checkpoint_dir, "best_loss_checkpoint.pth"
            )
            scheduler_state_dict = (
                scheduler.state_dict() if scheduler is not None else None
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

    logging.info("Training completed successfully.")
    logging.info(f"Final best loss: {best_loss:.6f}")
    logging.info(f"Total epochs trained: {num_epochs - start_epoch}")
