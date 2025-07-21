import logging
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from utils.checkpoint import load_checkpoint, save_checkpoint


def fit_model(
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    train_loader,
    num_epochs=10,
    device="cuda",
    mask_p=0.25,
    checkpoint_dir=".",
):
    logging.info(
        (
            f"Starting training - device: {device}, "
            f"epochs: {num_epochs}, mask_p: {mask_p}"
        )
    )
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
    model, optimizer, start_epoch, scheduler_state_dict, best_loss = (
        load_checkpoint(model, optimizer, latest_checkpoint, device)
    )

    # Restore scheduler state if it exists
    if scheduler is not None and scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)
        logging.info("Scheduler state restored from checkpoint")

    logging.info(f"Training starting from epoch: {start_epoch}")
    logging.info(f"Best loss so far: {best_loss:.6f}")

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training"):
        model.train()
        total_loss = 0
        batch_count = 0
        epoch_losses = []

        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Current learning rate: {current_lr:.6f}")

        for batch_idx, (x, y) in tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            total=len(train_loader),
        ):
            x = x.to(device)
            y = y.to(device)

            # Log batch information occasionally
            if batch_idx % 10 == 0:
                logging.debug(
                    (
                        f"Batch {batch_idx}/{len(train_loader)} - "
                        f"input shape: {x.shape}"
                    )
                )

            # forward pass
            x_recon = model(x)
            # loss
            loss = torch.sum((x_recon - y) * (x_recon - y)) / torch.sum(1)
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
