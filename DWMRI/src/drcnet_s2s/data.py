import logging

import numpy as np
import torch

np.random.seed(91021)


def sliding_windows(image, patch_size, step):
    """
    Memory-efficient chunked sliding window generation.

    Data layout: (Z, Vols, X, Y). Sliding is over z, x, y only (axis 1 is Vols).
    patch_size format: (z_patch, n_vols, x_patch, y_patch).
    """
    logging.info(
        f"Creating sliding windows (memory-efficient): patch_size={patch_size}, step={step}"
    )

    # Slide over axes 0, 2, 3 (z, x, y); skip axis 1 (Vols)
    spatial_patches_per_dim = [
        (image.shape[0] - patch_size[0]) // step + 1,
        (image.shape[2] - patch_size[2]) // step + 1,
        (image.shape[3] - patch_size[3]) // step + 1,
    ]
    total_spatial_patches = np.prod(spatial_patches_per_dim)

    logging.info(
        f"Will generate {total_spatial_patches:,} spatial patches with dimensions {spatial_patches_per_dim}"
    )
    logging.info(f"Each patch will contain all {patch_size[1]} volumes")

    # Process in chunks to avoid memory issues
    chunk_size = min(5000, total_spatial_patches)  # Process 5k patches at a time
    result_chunks = []

    idx = 0
    while idx < total_spatial_patches:
        end_idx = min(idx + chunk_size, total_spatial_patches)
        chunk_patches = end_idx - idx

        # Pre-allocate chunk: (chunk_patches, z_patch, n_vols, x_patch, y_patch)
        chunk = np.zeros((chunk_patches, *patch_size), dtype=image.dtype)

        # Fill chunk - iterate over z (axis 0), x (axis 2), y (axis 3)
        chunk_idx = 0
        for z in range(0, image.shape[0] - patch_size[0] + 1, step):
            for x in range(0, image.shape[2] - patch_size[2] + 1, step):
                for y in range(0, image.shape[3] - patch_size[3] + 1, step):
                    if idx <= chunk_idx < end_idx:
                        chunk[chunk_idx - idx] = image[
                            z : z + patch_size[0],
                            :,
                            x : x + patch_size[2],
                            y : y + patch_size[3],
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
    def __init__(self, data: np.ndarray, patch_size=32, step=16, mask_p=0.3):
        logging.info(
            f"Initializing DataSet: data.shape={data.shape}, patch_size={patch_size}, step={step}, mask_p={mask_p}"
        )
        # Transpose data from (X, Y, Z, Bvalues) to (Z, Bvalues, X, Y) i.e. (Z, Vols, X, Y)
        self.n_vols = data.shape[-1]
        data = np.transpose(data, (2, 3, 0, 1))
        logging.info(f"Data transposed to: {data.shape}")
        windows = sliding_windows(data, patch_size, step)
        self.windows = torch.from_numpy(windows).type(torch.float)
        logging.info(
            f"Created {self.windows.shape[0]} windows of shape {self.windows.shape[1:]}."
        )
        self.n_volumes = self.windows.shape[2]  # Vols is axis 2 in (N, Z, Vols, X, Y)

        # Each window will be analyzed n_vols times (once for each target volume)
        self.total_samples = len(self.windows) * self.n_volumes
        logging.info(
            f"Total training samples: {self.total_samples:,} (windows: {len(self.windows):,} × volumes: {self.n_volumes})"
        )
        # config params
        self.mask_p = mask_p

    def __getitem__(self, index: int):
        # Calculate which window and which volume to use as target
        window_idx = index // self.n_volumes
        target_volume_idx = index % self.n_volumes

        # S2S: single target volume with Bernoulli mask; model learns to denoise that volume.
        # One patch shape (Z, Vols, X, Y); take target volume -> (Z, X, Y)
        window_shape = (
            self.windows.shape[1],
            self.windows.shape[3],
            self.windows.shape[4],
        )  # (Z, X, Y)
        p_mtx = np.random.uniform(size=window_shape)
        mask = (p_mtx > self.mask_p).astype(np.double)
        mask = torch.tensor(mask, dtype=torch.float32)
        mask = mask.unsqueeze(0)  # (1, Z, X, Y)

        # Take target volume: windows[window_idx, :, target_volume_idx, :, :] -> (Z, X, Y)
        x_masked = self.windows[
            window_idx, :, target_volume_idx, :, :
        ].clone()
        x_masked = x_masked.unsqueeze(0) * mask  # (1, Z, X, Y)

        logging.debug(
            f"__getitem__ index={index}, window_idx={window_idx}, target_vol={target_volume_idx}, "
            f"x_masked.shape={x_masked.shape}"
        )
        return x_masked, mask

    def __len__(self):
        """Total number of training samples (windows × volumes)"""
        length = self.total_samples
        logging.debug(f"__len__ called, returning {length}")
        return length
