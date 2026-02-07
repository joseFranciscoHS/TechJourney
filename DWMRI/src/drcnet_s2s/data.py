"""
Training data for DRCNet-S2S (MD-S2S style).

Implements J-invariance at the pixel level: Bernoulli masks partition pixels into
visible (J^c) and occluded (J) sets. The network predicts occluded pixels using
only visible pixels from all volumes, exploiting both spatial and angular redundancy.
Data layout: (Vols, X, Y, Z).
"""
import logging

import numpy as np
import torch

np.random.seed(91021)


def sliding_windows(image, patch_size, step):
    """
    Memory-efficient chunked sliding window generation.
    Only applies sliding windows to spatial dimensions (x, y, z), not volume dimension.
    """
    logging.info(
        f"Creating sliding windows (memory-efficient): patch_size={patch_size}, step={step}"
    )

    # Calculate number of patches per SPATIAL dimension only (skip volume dimension)
    # patch_size format: (volumes, x, y, z)
    # We only slide over spatial dimensions (x, y, z)
    spatial_patches_per_dim = []
    for i in range(1, len(patch_size)):  # Skip volume dimension (i=0)
        patches = (image.shape[i] - patch_size[i]) // step + 1
        spatial_patches_per_dim.append(patches)

    total_spatial_patches = np.prod(spatial_patches_per_dim)

    logging.info(
        f"Will generate {total_spatial_patches:,} spatial patches with dimensions {spatial_patches_per_dim}"
    )
    logging.info(f"Each patch will contain all {patch_size[0]} volumes")

    # Process in chunks to avoid memory issues
    chunk_size = min(5000, total_spatial_patches)  # Process 5k patches at a time
    result_chunks = []

    idx = 0
    while idx < total_spatial_patches:
        end_idx = min(idx + chunk_size, total_spatial_patches)
        chunk_patches = end_idx - idx

        # Pre-allocate chunk
        chunk = np.zeros((chunk_patches, *patch_size), dtype=image.dtype)

        # Fill chunk - only iterate over spatial dimensions
        chunk_idx = 0
        for x in range(0, image.shape[1] - patch_size[1] + 1, step):
            for y in range(0, image.shape[2] - patch_size[2] + 1, step):
                for z in range(0, image.shape[3] - patch_size[3] + 1, step):
                    if idx <= chunk_idx < end_idx:
                        # Take all volumes for this spatial location
                        chunk[chunk_idx - idx] = image[
                            :,
                            x : x + patch_size[1],
                            y : y + patch_size[2],
                            z : z + patch_size[3],
                        ]
                    chunk_idx += 1
                    if chunk_idx >= end_idx:
                        break
                if chunk_idx >= end_idx:
                    break
            if chunk_idx >= end_idx:
                break

        result_chunks.append(chunk)
        idx = end_idx
        logging.debug(f"Processed chunk: {chunk.shape[0]} patches")

    # Concatenate all chunks
    result = np.concatenate(result_chunks, axis=0)
    logging.info(f"Sliding windows generated: {result.shape}")
    return result


class TrainingDataSet(torch.utils.data.Dataset):
    """
    Dataset for MD-S2S style training with Bernoulli pixel masking.

    Applies J-invariance: each sample is a 4D patch (Vols, X, Y, Z) with a random
    Bernoulli mask. The network receives masked input and learns to predict occluded
    pixels from visible ones. Loss is computed only on masked pixels (see fit.py).
    """

    def __init__(self, data: np.ndarray, patch_size=32, step=16, mask_p=0.3):
        logging.info(
            f"Initializing DataSet: data.shape={data.shape}, patch_size={patch_size}, step={step}, mask_p={mask_p}"
        )
        # Transpose data from (X, Y, Z, Bvalues) to (Vols, X, Y, Z)
        self.n_vols = data.shape[-1]
        data = np.transpose(data, (3, 0, 1, 2))
        logging.info(f"Data transposed to: {data.shape}")
        windows = sliding_windows(data, patch_size, step)
        self.windows = torch.from_numpy(windows).type(torch.float)
        logging.info(
            f"Created {self.windows.shape[0]} windows of shape {self.windows.shape[1:]}."
        )
        self.n_volumes = self.windows.shape[1]  # Vols is axis 2 in (N, Z, Vols, X, Y)

        # Each window will be analyzed once
        self.total_samples = len(self.windows)
        logging.info(
            f"Total training samples: {self.total_samples:,} (windows: {len(self.windows):,})"
        )
        # config params
        self.mask_p = mask_p

    def __getitem__(self, index: int):
        """Return masked input (x_masked) and mask for J-invariant loss.
        mask=1 for visible pixels, mask=0 for occluded (J dimensions)."""
        # Calculate which window to use as target
        window_idx = index

        # S2S: target volumes with Bernoulli mask;
        # One patch shape (Vols, X, Y, Z)
        window_shape = (
            self.windows.shape[1],
            self.windows.shape[2],
            self.windows.shape[3],
            self.windows.shape[4],
        )  # (Vols, X, Y, Z)
        p_mtx = np.random.uniform(size=window_shape)
        mask = (p_mtx > self.mask_p).astype(np.double)
        mask = torch.tensor(mask, dtype=torch.float32)  # (Vols, X, Y, Z)

        # Take all volumes: windows[window_idx, :, :, :, :] -> (Vols, X, Y, Z)
        x_masked = self.windows[window_idx].clone()
        x_masked = x_masked * mask  # (Vols, X, Y, Z)

        logging.debug(
            f"__getitem__ index={index}, window_idx={window_idx}, "
            f"x_masked.shape={x_masked.shape}"
        )
        return x_masked, mask

    def __len__(self):
        """Total number of training samples (windows × volumes)"""
        length = self.total_samples
        logging.debug(f"__len__ called, returning {length}")
        return length
