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
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "loss": loss,
        "best_loss": best_loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename, device="cuda"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        best_loss = checkpoint["best_loss"]
        scheduler_state_dict = checkpoint.get("scheduler_state_dict", None)
        print(
            f"Checkpoint loaded: {filename} | loss : {loss} | best_loss : {best_loss}"
        )
        return model, optimizer, epoch, scheduler_state_dict, best_loss
    else:
        print(f"No checkpoint found at {filename}")
        return model, optimizer, 0, None, float("inf")
