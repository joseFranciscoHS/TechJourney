import logging

import numpy as np
import torch
from tqdm import tqdm


def reconstruct_dwis(model, data, device, mask_p=0.3, n_preds=10):
    """
    Reconstruct full-size DWI data using hybrid MD-S2S approach.

    Uses the same masking strategy as training: all volumes are included,
    but the target volume is masked with Bernoulli mask. Multiple predictions
    with different masks are averaged for robustness.

    Args:
        model: Trained DRCNet-hybrid model
        data: Input data of shape (Vols, X, Y, Z) - full-size data
        device: Device to run inference on
        mask_p: Mask probability (same as training)
        n_preds: Number of predictions per volume (for averaging with different masks)

    Returns:
        Reconstructed data of shape (Vols, X, Y, Z)
    """
    logging.info(f"Starting DWI reconstruction on device: {device}")
    logging.info(f"Input data shape: {data.shape}")
    logging.info(
        f"Using hybrid MD-S2S approach with mask_p={mask_p}, n_preds={n_preds}"
    )

    model.to(device)
    model.eval()

    num_vols, x_size, y_size, z_size = data.shape
    spatial_dims = (x_size, y_size, z_size)

    # Initialize output array
    sum_preds = np.zeros((num_vols, *spatial_dims), dtype=np.float32)

    with torch.inference_mode():
        data_device = data.to(device)

        # Process each target volume separately
        for vol_idx in tqdm(range(num_vols), desc="Processing volumes"):
            logging.info(f"Processing volume {vol_idx + 1}/{num_vols}")

            # Multiple predictions per volume for robustness (different random masks)
            for pred_idx in range(n_preds):
                # Create masked input (same as training approach)
                # Generate random mask for the target volume
                p_mtx = np.random.uniform(size=spatial_dims)
                mask = (p_mtx > mask_p).astype(np.float32)
                mask_tensor = (
                    torch.tensor(mask).to(device, dtype=torch.float32).unsqueeze(0)
                )

                # Apply mask to target volume only (keep all volumes in input)
                data_masked = data_device.clone()
                data_masked[vol_idx] = data_device[vol_idx] * mask_tensor

                # Add batch dimension: (1, num_vols, X, Y, Z)
                data_masked = data_masked.unsqueeze(0)

                # Forward pass: model expects (B, C, X, Y, Z)
                reconstructed = model(data_masked)

                # Extract the target volume prediction
                # reconstructed shape: (1, 1, X, Y, Z)
                pred_volume = reconstructed.squeeze(0).squeeze(0).detach().cpu().numpy()

                # Accumulate predictions
                sum_preds[vol_idx] += pred_volume

        # Average predictions
        reconstructed = sum_preds / n_preds

        logging.info(f"Reconstruction completed. Output shape: {reconstructed.shape}")
        logging.info(
            f"Output stats - Min: {reconstructed.min():.4f}, "
            f"Max: {reconstructed.max():.4f}, Mean: {reconstructed.mean():.4f}"
        )

    return reconstructed
