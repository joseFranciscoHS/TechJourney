import logging

import numpy as np
import torch
from tqdm import tqdm


def reconstruct_dwis(model, data, device, mask_p=0.3, n_preds=10):
    """
    Reconstruct full-size DWI data using DRCNet with S2S framework.

    All volumes at a time, masked with Bernoulli mask (same as training).
    Multiple predictions with different masks are averaged for robustness.

    Args:
        model: Trained DRCNet-S2S model
        data: Input data of shape (Z, Vols, X, Y) - full-size data
        device: Device to run inference on
        mask_p: Mask probability (same as training)
        n_preds: Number of predictions per volume (for averaging with different masks)

    Returns:
        Reconstructed data of shape (Z, Vols, X, Y)
    """
    logging.info(f"Starting DWI reconstruction on device: {device}")
    logging.info(f"Input data shape: {data.shape}")
    logging.info(f"Using S2S framework with mask_p={mask_p}, n_preds={n_preds}")

    model.to(device)
    model.eval()

    z_size, num_vols, x_size, y_size = data.shape
    spatial_dims = (z_size, x_size, y_size)

    # Initialize output array (num_vols, Z, X, Y)
    sum_preds = np.zeros((num_vols, *spatial_dims), dtype=np.float32)

    with torch.inference_mode():
        data_device = data.to(device)

        # Process each target volume separately
        for pred_idx in range(n_preds):
            # Create masked input (same as training approach)
            # Generate random mask for the target volumes (Z, Vols, X, Y)
            p_mtx = np.random.uniform(size=(z_size, num_vols, x_size, y_size))
            mask = (p_mtx > mask_p).astype(np.float32)
            mask_tensor = torch.tensor(mask).to(device, dtype=torch.float32)
            mask_tensor = mask_tensor.unsqueeze(0)  # (1, Z, Vols, X, Y)

            # data_masked: (Z, Vols, X, Y); add channel and batch -> (1, Z, Vols, X, Y)
            data_masked = data_device.clone()
            data_masked = data_masked.unsqueeze(0) * mask_tensor  # (1, Z, Vols, X, Y)

            # Forward pass: model expects (B, Z, Vols, X, Y)
            reconstructed = model(data_masked)

            # reconstructed shape: (1, 1, Z, Vols, X, Y)
            pred_volume = reconstructed.squeeze(0).squeeze(0).detach().cpu().numpy()

            # Accumulate predictions
            sum_preds += pred_volume

        # Average predictions; transpose (num_vols, Z, X, Y) -> (Z, Vols, X, Y)
        reconstructed = np.transpose(sum_preds / n_preds, (1, 0, 2, 3))

        logging.info(f"Reconstruction completed. Output shape: {reconstructed.shape}")
        logging.info(
            f"Output stats - Min: {reconstructed.min():.4f}, "
            f"Max: {reconstructed.max():.4f}, Mean: {reconstructed.mean():.4f}"
        )

    return reconstructed
