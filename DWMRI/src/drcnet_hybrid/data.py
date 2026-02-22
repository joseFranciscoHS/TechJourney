import logging

import numpy as np
import torch

np.random.seed(91021)


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
        
        # transpose data from (X, Y, Z, Bvalues) to (Bvalues, X, Y, Z)
        self.n_vols = data.shape[-1]
        data_transposed = np.transpose(data, (3, 0, 1, 2))
        logging.info(f"Noisy data transposed to: {data_transposed.shape}")
        
        # Transpose clean_data if provided
        clean_data_transposed = None
        if clean_data is not None:
            clean_data_transposed = np.transpose(clean_data, (3, 0, 1, 2))
            logging.info(f"Clean data transposed to: {clean_data_transposed.shape}")
        
        # Handle patch_size as int or tuple
        if isinstance(patch_size, int):
            patch_size_tuple = (self.n_vols, patch_size, patch_size, patch_size)
        else:
            patch_size_tuple = patch_size
        
        windows = sliding_windows(
            data_transposed,
            patch_size_tuple,
            step,
            clean_data=clean_data_transposed,
            mask=brain_mask,
            patch_filter_method=patch_filter_method,
            min_signal_threshold=min_signal_threshold,
        )
        self.windows = torch.from_numpy(windows).type(torch.float)
        logging.info(
            f"Created {self.windows.shape[0]} windows of shape {self.windows.shape[1:]}."
        )
        self.n_volumes = self.windows.shape[1]

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

        # Hybrid MD-S2S approach: include target volume with Bernoulli mask
        # generating the mask for the spatial dimensions
        window_shape = list(self.windows.shape[2:])
        p_mtx = np.random.uniform(size=window_shape)
        mask = (p_mtx > self.mask_p).astype(np.double)
        mask = torch.tensor(mask, dtype=torch.float32)
        # adding a channel dimension to the mask
        mask = mask.unsqueeze(0)

        # Create input by taking all volumes INCLUDING the target (with mask applied)
        x_masked = self.windows[window_idx].clone()
        # only mask the target volume
        volume_masked = x_masked[target_volume_idx] * mask
        x_masked[target_volume_idx] = volume_masked
        # original noisy target volume
        noisy_target_volume = self.windows[
            window_idx, target_volume_idx : target_volume_idx + 1
        ]
        logging.debug(
            f"__getitem__ index={index}, window_idx={window_idx}, target_vol={target_volume_idx}, "
            f"x_masked.shape={x_masked.shape}, noisy_target_volume.shape={noisy_target_volume.shape}"
        )
        return x_masked, mask, noisy_target_volume

    def __len__(self):
        """Total number of training samples (windows × volumes)"""
        length = self.total_samples
        logging.debug(f"__len__ called, returning {length}")
        return length
