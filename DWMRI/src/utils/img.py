import logging

import matplotlib.pyplot as plt


def save_volume_image(volume, slice_idx=None, file_name=""):
    logging.debug(
        f"Saving volume image - volume shape: {volume.shape}, slice_idx: {slice_idx}, file_name: {file_name}"
    )

    if slice_idx is None:
        slice_idx = volume.shape[0] // 2
        logging.debug(f"Using middle slice: {slice_idx}")

    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    axes.imshow(volume[slice_idx], cmap="gray")
    axes.axis("off")
    plt.tight_layout()

    if file_name:
        try:
            plt.savefig(file_name)
            logging.info(f"Volume image saved to: {file_name}")
        except Exception as e:
            logging.error(f"Failed to save volume image: {e}")

    plt.show()
    logging.debug("Volume image displayed")


def visualize_single_volume(
    original_volume,
    denoised_volume,
    slice_idx=None,
    title1="Original",
    title2="Denoised",
    file_name="",
):
    logging.debug(
        f"Visualizing single volume - original shape: {original_volume.shape}, denoised shape: {denoised_volume.shape}"
    )
    logging.debug(
        f"Visualization params - slice_idx: {slice_idx}, title1: {title1}, title2: {title2}, file_name: {file_name}"
    )

    if slice_idx is None:
        slice_idx = original_volume.shape[0] // 2  # Middle slice by default
        logging.debug(f"Using middle slice: {slice_idx}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"{title1} vs {title2} Volume (Slice {slice_idx})")

    axes[0].imshow(original_volume[slice_idx], cmap="gray")
    axes[0].set_title(title1)
    axes[0].axis("off")

    axes[1].imshow(denoised_volume[slice_idx], cmap="gray")
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.tight_layout()

    if file_name:
        try:
            plt.savefig(file_name)
            logging.info(f"Volume comparison saved to: {file_name}")
        except Exception as e:
            logging.error(f"Failed to save volume comparison: {e}")

    plt.show()
    logging.debug("Volume comparison displayed")
    logging.debug("Volume comparison displayed")
