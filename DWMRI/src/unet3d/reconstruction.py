import logging

import numpy as np
import torch
from tqdm import tqdm


def reconstruct_dwis(model, data, device, mask_p=0.3, n_preds=10):
    logging.info(f"Starting reconstruction on device: {device}")
    model.to(device)
    model.eval()
    with torch.inference_mode():
        sum_preds = np.zeros(data.shape)
        logging.info(f"Sum preds shape: {sum_preds.shape}")
        x = data.to(device)
        logging.info(f"X shape: {x.shape}")
        num_vols = x.shape[0]
        logging.info(f"Num vols: {num_vols}")
        spatial_dims = x.shape[1:]
        logging.info(f"Spatial dims: {spatial_dims}")
        for i in tqdm(range(num_vols), desc="Reconstructing"):
            idxs = list(range(i, (i + 1)))
            for j in range(n_preds):
                p_mtx = np.random.uniform(size=spatial_dims)
                mask = (p_mtx > mask_p).astype(np.double)
                mask = torch.tensor(mask).to(device, dtype=torch.float32).unsqueeze(0)

                volume_masked = x[i, :, :, :] * mask
                x_masked = x.clone()
                x_masked[i] = volume_masked

                reconstructed = model(x_masked.unsqueeze(0))
                sum_preds[idxs, :, :, :] += (
                    reconstructed.squeeze().detach().cpu().numpy()
                )

        sum_preds = sum_preds / n_preds

    return np.array(sum_preds)
