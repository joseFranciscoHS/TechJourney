import logging
import os

import torch


def save_checkpoint(
    model_state_dict,
    optimizer_state_dict,
    epoch,
    loss,
    best_loss,
    filename,
    scheduler_state_dict,
):
    logging.info(f"Saving checkpoint to: {filename}")
    logging.debug(
        f"Checkpoint details - epoch: {epoch}, loss: {loss:.6f}, best_loss: {best_loss:.6f}"
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "loss": loss,
        "best_loss": best_loss,
    }

    try:
        torch.save(checkpoint, filename)
        logging.info(f"Checkpoint saved successfully: {filename}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")
        raise


def load_checkpoint(model, optimizer, filename, device="cuda", strict=True):
    logging.info(f"Attempting to load checkpoint from: {filename}")

    if os.path.isfile(filename):
        try:
            checkpoint = torch.load(
                filename, map_location=torch.device(device)
            )
            
            if model is not None:
                model_state_dict = checkpoint["model_state_dict"]
                
                # Handle DataParallel state dict (remove 'module.' prefix)
                if any(key.startswith('module.') for key in model_state_dict.keys()):
                    logging.info("Detected DataParallel checkpoint, removing 'module.' prefix")
                    model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
                
                # Load with strict=False to handle architecture changes gracefully
                missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=strict)
                
                if missing_keys:
                    logging.warning(f"Missing keys in checkpoint: {len(missing_keys)} keys")
                    if len(missing_keys) <= 10:  # Log first 10 missing keys
                        logging.warning(f"Missing keys: {missing_keys}")
                    else:
                        logging.warning(f"First 10 missing keys: {missing_keys[:10]}")
                
                if unexpected_keys:
                    logging.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                    if len(unexpected_keys) <= 10:  # Log first 10 unexpected keys
                        logging.warning(f"Unexpected keys: {unexpected_keys}")
                    else:
                        logging.warning(f"First 10 unexpected keys: {unexpected_keys[:10]}")
            
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            epoch = checkpoint["epoch"]
            loss = checkpoint["loss"]
            best_loss = checkpoint["best_loss"]
            scheduler_state_dict = checkpoint.get("scheduler_state_dict", None)

            logging.info(f"Checkpoint loaded successfully: {filename}")
            logging.info(
                f"Checkpoint details - epoch: {epoch}, loss: {loss:.6f}, best_loss: {best_loss:.6f}"
            )
            if scheduler_state_dict is not None:
                logging.info("Scheduler state found in checkpoint")
            else:
                logging.info("No scheduler state in checkpoint")

            return model, optimizer, epoch, scheduler_state_dict, best_loss
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            logging.info("Starting training from scratch")
            return model, optimizer, 0, None, float("inf")
    else:
        logging.info(
            f"No checkpoint found at {filename}, starting training from scratch"
        )
        return model, optimizer, 0, None, float("inf")
