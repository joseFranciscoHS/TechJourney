import logging

import numpy as np
import torch
from tqdm import tqdm


def reconstruct_dwis(model, data_loader, device):
    logging.info(f"Starting DWI reconstruction on device: {device}")
    model.to(device)
    model.eval()
    reconstructed_dwis = []
    with torch.inference_mode():
        for x,_ in tqdm(data_loader, desc="Reconstructing"):
            x = x.to(device)
            reconstructed = model(x)
            reconstructed_dwis.append(reconstructed.squeeze().detach().cpu().numpy())
    logging.info("DWI reconstruction completed.")
    return np.array(reconstructed_dwis)
