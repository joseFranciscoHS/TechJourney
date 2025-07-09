import math

import numpy as np
from skimage.metrics import structural_similarity


def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    max_value = np.max(original)
    psnr_value = 20 * math.log10(max_value / math.sqrt(mse))
    return psnr_value


def ssim(original, reconstructed):
    ssim_value = structural_similarity(
        original,
        reconstructed,
        data_range=original.max() - original.min(),
        channel_axis=3,
    )
    return ssim_value or 0


def mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)
