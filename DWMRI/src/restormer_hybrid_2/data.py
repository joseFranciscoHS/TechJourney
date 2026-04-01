import logging

import numpy as np
import torch

np.random.seed(91021)


def compute_valid_patch_coords(
    image,
    patch_size,
    step,
    clean_data=None,
    mask=None,
    patch_filter_method="none",
    min_signal_threshold=0.0,
):
    """
    First pass only: compute valid (x, y, z) coordinates for sliding windows
    with optional patch filtering. Does not allocate the full patch array.

    Args:
        image: 4D array (volumes, X, Y, Z)
        patch_size: tuple (volumes, patch_x, patch_y, patch_z)
        step: step size for sliding window
        clean_data: optional 4D array (volumes, X, Y, Z) for threshold filtering
        mask: optional 3D boolean array (X, Y, Z) for otsu filtering
        patch_filter_method: "none", "threshold", or "otsu"
        min_signal_threshold: threshold for "threshold" method

    Returns:
        valid_coords: list of (x, y, z)
        patch_size: same tuple as input (for convenience)
    """
    logging.info(
        f"Creating sliding windows: patch_size={patch_size}, step={step}, "
        f"filter_method={patch_filter_method}, threshold={min_signal_threshold}"
    )

    spatial_patches_per_dim = []
    for i in range(1, len(patch_size)):
        patches = (image.shape[i] - patch_size[i]) // step + 1
        spatial_patches_per_dim.append(patches)

    total_spatial_patches = np.prod(spatial_patches_per_dim)

    logging.info(
        f"Potential patches: {total_spatial_patches:,} with dimensions {spatial_patches_per_dim}"
    )
    logging.info(f"Each patch will contain all {patch_size[0]} volumes")

    valid_coords = []
    skipped_count = 0

    for x in range(0, image.shape[1] - patch_size[1] + 1, step):
        for y in range(0, image.shape[2] - patch_size[2] + 1, step):
            for z in range(0, image.shape[3] - patch_size[3] + 1, step):
                include_patch = True

                if patch_filter_method == "threshold" and clean_data is not None:
                    clean_patch = clean_data[
                        :,
                        x : x + patch_size[1],
                        y : y + patch_size[2],
                        z : z + patch_size[3],
                    ]
                    if clean_patch.max() <= min_signal_threshold:
                        include_patch = False

                elif patch_filter_method == "otsu" and mask is not None:
                    mask_patch = mask[
                        x : x + patch_size[1],
                        y : y + patch_size[2],
                        z : z + patch_size[3],
                    ]
                    if mask_patch.sum() == 0:
                        include_patch = False

                if include_patch:
                    valid_coords.append((x, y, z))
                else:
                    skipped_count += 1

    num_valid_patches = len(valid_coords)

    if patch_filter_method != "none":
        logging.info(
            f"Patch filtering: kept {num_valid_patches:,} patches, "
            f"skipped {skipped_count:,} ({100*skipped_count/total_spatial_patches:.1f}%)"
        )

    return valid_coords, patch_size


def sliding_windows(
    image,
    patch_size,
    step,
    clean_data=None,
    mask=None,
    patch_filter_method="none",
    min_signal_threshold=0.0,
):
    """
    Memory-efficient chunked sliding window generation with optional patch filtering.
    Only applies sliding windows to spatial dimensions (x, y, z), not volume dimension.
    
    Args:
        image: 4D array (volumes, X, Y, Z) - the noisy data to create patches from
        patch_size: tuple (volumes, patch_x, patch_y, patch_z)
        step: step size for sliding window
        clean_data: optional 4D array (volumes, X, Y, Z) - clean data for threshold filtering
        mask: optional 3D boolean array (X, Y, Z) - brain mask for otsu filtering
        patch_filter_method: "none", "threshold", or "otsu"
        min_signal_threshold: threshold for "threshold" method (exclude if max <= threshold)
    
    Returns:
        result: array of patches (N, volumes, patch_x, patch_y, patch_z)
    """
    logging.info(
        f"Creating sliding windows: patch_size={patch_size}, step={step}, "
        f"filter_method={patch_filter_method}, threshold={min_signal_threshold}"
    )

    # Calculate number of patches per SPATIAL dimension only (skip volume dimension)
    spatial_patches_per_dim = []
    for i in range(1, len(patch_size)):  # Skip volume dimension (i=0)
        patches = (image.shape[i] - patch_size[i]) // step + 1
        spatial_patches_per_dim.append(patches)

    total_spatial_patches = np.prod(spatial_patches_per_dim)

    logging.info(
        f"Potential patches: {total_spatial_patches:,} with dimensions {spatial_patches_per_dim}"
    )
    logging.info(f"Each patch will contain all {patch_size[0]} volumes")

    # First pass: collect valid patch coordinates
    valid_coords = []
    skipped_count = 0
    
    for x in range(0, image.shape[1] - patch_size[1] + 1, step):
        for y in range(0, image.shape[2] - patch_size[2] + 1, step):
            for z in range(0, image.shape[3] - patch_size[3] + 1, step):
                include_patch = True
                
                if patch_filter_method == "threshold" and clean_data is not None:
                    clean_patch = clean_data[
                        :,
                        x : x + patch_size[1],
                        y : y + patch_size[2],
                        z : z + patch_size[3],
                    ]
                    if clean_patch.max() <= min_signal_threshold:
                        include_patch = False
                        
                elif patch_filter_method == "otsu" and mask is not None:
                    mask_patch = mask[
                        x : x + patch_size[1],
                        y : y + patch_size[2],
                        z : z + patch_size[3],
                    ]
                    if mask_patch.sum() == 0:
                        include_patch = False
                
                if include_patch:
                    valid_coords.append((x, y, z))
                else:
                    skipped_count += 1

    num_valid_patches = len(valid_coords)
    
    if patch_filter_method != "none":
        logging.info(
            f"Patch filtering: kept {num_valid_patches:,} patches, "
            f"skipped {skipped_count:,} ({100*skipped_count/total_spatial_patches:.1f}%)"
        )
    
    if num_valid_patches == 0:
        logging.warning("No valid patches found after filtering! Returning empty array.")
        return np.zeros((0, *patch_size), dtype=image.dtype)

    # Second pass: create patches for valid coordinates in chunks
    chunk_size = min(5000, num_valid_patches)
    result_chunks = []

    for start_idx in range(0, num_valid_patches, chunk_size):
        end_idx = min(start_idx + chunk_size, num_valid_patches)
        chunk_coords = valid_coords[start_idx:end_idx]
        chunk_patches = len(chunk_coords)
        
        chunk = np.zeros((chunk_patches, *patch_size), dtype=image.dtype)
        
        for i, (x, y, z) in enumerate(chunk_coords):
            chunk[i] = image[
                :,
                x : x + patch_size[1],
                y : y + patch_size[2],
                z : z + patch_size[3],
            ]
        
        result_chunks.append(chunk)
        logging.debug(f"Processed chunk: {chunk.shape[0]} patches")

    result = np.concatenate(result_chunks, axis=0)
    logging.info(f"Sliding windows generated: {result.shape}")
    return result


class TrainingDataSet(torch.utils.data.Dataset):
    """
    Lazy training dataset: stores only valid patch coordinates and slices
    from the 4D volume in __getitem__, so the full patch array is never
    materialized in RAM.
    """

    def __init__(
        self,
        data: np.ndarray,
        patch_size=32,
        step=16,
        mask_p=0.3,
        clean_data: np.ndarray = None,
        brain_mask: np.ndarray = None,
        patch_filter_method: str = "none",
        min_signal_threshold: float = 0.0,
    ):
        """
        Training dataset for DWMRI denoising with optional patch filtering.

        Args:
            data: noisy 4D array (X, Y, Z, volumes)
            patch_size: spatial patch size (or tuple)
            step: step size for sliding windows
            mask_p: Bernoulli mask probability for hybrid MD-S2S training
            clean_data: optional clean 4D array (X, Y, Z, volumes) for threshold filtering
            brain_mask: optional 3D boolean array (X, Y, Z) for otsu filtering
            patch_filter_method: "none", "threshold", or "otsu"
            min_signal_threshold: threshold for "threshold" method
        """
        logging.info(
            f"Initializing DataSet: data.shape={data.shape}, patch_size={patch_size}, "
            f"step={step}, mask_p={mask_p}, filter_method={patch_filter_method}"
        )

        # Transpose data from (X, Y, Z, volumes) to (volumes, X, Y, Z); keep reference only
        self.n_vols = data.shape[-1]
        self.data_transposed = np.transpose(data, (3, 0, 1, 2))
        logging.info(f"Noisy data transposed to: {self.data_transposed.shape}")

        clean_data_transposed = None
        if clean_data is not None:
            clean_data_transposed = np.transpose(clean_data, (3, 0, 1, 2))
            logging.info(f"Clean data transposed to: {clean_data_transposed.shape}")

        if isinstance(patch_size, int):
            patch_size_tuple = (self.n_vols, patch_size, patch_size, patch_size)
        else:
            patch_size_tuple = patch_size

        self.valid_coords, self.patch_size_tuple = compute_valid_patch_coords(
            self.data_transposed,
            patch_size_tuple,
            step,
            clean_data=clean_data_transposed,
            mask=brain_mask,
            patch_filter_method=patch_filter_method,
            min_signal_threshold=min_signal_threshold,
        )

        if len(self.valid_coords) == 0:
            raise ValueError(
                "No valid patches found after filtering. Cannot create empty dataset."
            )

        self.n_volumes = self.n_vols
        self.total_samples = len(self.valid_coords) * self.n_volumes
        self.mask_p = mask_p

        logging.info(
            f"Lazy dataset: {len(self.valid_coords):,} valid patches, "
            f"total_samples={self.total_samples:,} (patches × volumes)"
        )

    def __getitem__(self, index: int):
        window_idx = index // self.n_volumes
        target_volume_idx = index % self.n_volumes

        x, y, z = self.valid_coords[window_idx]
        px, py, pz = (
            self.patch_size_tuple[1],
            self.patch_size_tuple[2],
            self.patch_size_tuple[3],
        )

        window = self.data_transposed[
            :, x : x + px, y : y + py, z : z + pz
        ].copy()
        window = torch.from_numpy(window).float()

        # Bernoulli mask for spatial dimensions (hybrid MD-S2S)
        window_shape = list(window.shape[1:])
        p_mtx = np.random.uniform(size=window_shape)
        mask = (p_mtx > self.mask_p).astype(np.double)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        x_masked = window.clone()
        volume_masked = x_masked[target_volume_idx] * mask.squeeze(0)
        x_masked[target_volume_idx] = volume_masked
        noisy_target_volume = window[target_volume_idx : target_volume_idx + 1]

        logging.debug(
            f"__getitem__ index={index}, window_idx={window_idx}, target_vol={target_volume_idx}, "
            f"x_masked.shape={x_masked.shape}, noisy_target_volume.shape={noisy_target_volume.shape}"
        )
        return x_masked, mask, noisy_target_volume

    def __len__(self):
        return self.total_samples
