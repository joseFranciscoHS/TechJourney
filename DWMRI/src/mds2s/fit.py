import logging
import os

import numpy as np
import torch
from tqdm import tqdm
from utlis.checkpoint import load_checkpoint, save_checkpoint


def fit_model(
    model,
    optimizer,
    scheduler,
    train_loader,
    num_epochs=10,
    num_volumes=6,
    device="cuda",
    mask_p=0.25,
    checkpoint_dir="",
):
    model.to(device)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load the latest checkpoint if it exists
    latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    model, optimizer, start_epoch, scheduler_state_dict, best_loss = load_checkpoint(
        model, optimizer, latest_checkpoint
    )

    # Restore scheduler state if it exists
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    logging.info(f"start epoch : {start_epoch}")

    for epoch in tqdm(
        range(start_epoch, num_epochs), desc=f"Training volumes {num_volumes}"
    ):
        model.train()
        total_loss = 0

        for x in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            [x] = x
            x = x.to(device)
            p_mtx = np.random.uniform(
                size=[x.shape[0], x.shape[1], x.shape[2], x.shape[3]]
            )
            mask = (p_mtx > mask_p).astype(np.double)
            mask = torch.tensor(mask).to(device, dtype=torch.float32)

            x_masked = x * mask
            # forward pass
            x_recon = model(x_masked)
            # loss
            loss = torch.sum((x_recon - x) * (x_recon - x) * (1 - mask)) / torch.sum(
                1 - mask
            )
            # zero grad
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # step
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        scheduler.step(avg_loss)

        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        logging.info(f"Average Loss: {avg_loss:.6f}")

        # Save the latest checkpoint
        save_checkpoint(
            model.state_dict(),
            optimizer.state_dict(),
            epoch + 1,
            avg_loss,
            best_loss,
            latest_checkpoint,
            scheduler.state_dict(),
        )

    logging.info("Training completed.")
