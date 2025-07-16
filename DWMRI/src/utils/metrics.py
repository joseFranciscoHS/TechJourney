import logging
import math

import numpy as np
from skimage.metrics import structural_similarity


def psnr(original, reconstructed):
    logging.debug(
        f"Computing PSNR - original shape: {original.shape}, reconstructed shape: {reconstructed.shape}"
    )

    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        logging.debug("MSE is 0, returning infinite PSNR")
        return float("inf")

    max_value = np.max(original)
    psnr_value = 20 * math.log10(max_value / math.sqrt(mse))

    logging.debug(
        f"PSNR computed - MSE: {mse:.6f}, Max value: {max_value:.4f}, PSNR: {psnr_value:.4f}"
    )
    return psnr_value


def ssim(original, reconstructed):
    logging.debug(
        f"Computing SSIM - original shape: {original.shape}, reconstructed shape: {reconstructed.shape}"
    )

    data_range = original.max() - original.min()
    logging.debug(f"SSIM data range: {data_range:.4f}")

    try:
        ssim_value = structural_similarity(
            original,
            reconstructed,
            data_range=data_range,
            channel_axis=3,
        )
        logging.debug(f"SSIM computed: {ssim_value:.4f}")
        return ssim_value or 0
    except Exception as e:
        logging.error(f"Error computing SSIM: {e}")
        return 0


def mse(original, reconstructed):
    logging.debug(
        f"Computing MSE - original shape: {original.shape}, reconstructed shape: {reconstructed.shape}"
    )

    mse_value = np.mean((original - reconstructed) ** 2)
    logging.debug(f"MSE computed: {mse_value:.6f}")
    return mse_value
