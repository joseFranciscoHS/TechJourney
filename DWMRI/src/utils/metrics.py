import json
import logging
import math
import os

import matplotlib.pyplot as plt
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
    return float(psnr_value)


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
        return float(ssim_value) or 0
    except Exception as e:
        logging.error(f"Error computing SSIM: {e}")
        return 0


def mse(original, reconstructed):
    logging.debug(
        f"Computing MSE - original shape: {original.shape}, reconstructed shape: {reconstructed.shape}"
    )

    mse_value = np.mean((original - reconstructed) ** 2)
    logging.debug(f"MSE computed: {mse_value:.6f}")
    return float(mse_value)


def compute_metrics(original, reconstructed, metrics=["psnr", "ssim", "mse"]):
    metrics_values = {}
    if "psnr" in metrics:
        metrics_values["psnr"] = psnr(original, reconstructed)
    if "ssim" in metrics:
        metrics_values["ssim"] = ssim(original, reconstructed)
    if "mse" in metrics:
        metrics_values["mse"] = mse(original, reconstructed)
    return metrics_values


def save_metrics(metrics, metrics_dir):
    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


def compare_volumes(
    original_volume,
    denoised_volume,
    volume_idx=0,
    slice_idx=None,
    file_name="",
):
    if slice_idx is None:
        slice_idx = original_volume.shape[0] // 2  # Middle slice by default

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle(f"Original vs Denoised Volume (Slice {slice_idx})")

    axes[0].imshow(original_volume[slice_idx, volume_idx, :, :], cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(denoised_volume[slice_idx, volume_idx, :, :], cmap="gray")
    axes[1].set_title("Denoised")
    axes[1].axis("off")

    rms_diff = np.sqrt(
        (
            original_volume[slice_idx, volume_idx, :, :]
            - denoised_volume[slice_idx, volume_idx, :, :]
        )
        ** 2
    )
    axes[2].imshow(rms_diff, cmap="gray")
    axes[2].set_title("RMS Diff")
    axes[2].axis("off")

    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, bbox_inches="tight")
    plt.show()


def visualize_single_volume(
    original_volume,
    volume_idx=0,
    slice_idx=None,
    file_name="",
):
    if slice_idx is None:
        print(original_volume.shape)
        slice_idx = original_volume.shape[0] // 2  # Middle slice by default

    plt.imshow(original_volume[slice_idx, volume_idx, :, :], cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, bbox_inches="tight")
    plt.show()
