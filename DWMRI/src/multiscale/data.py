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
    
    logging.info(f"Will generate {total_spatial_patches:,} spatial patches with dimensions {spatial_patches_per_dim}")
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
                        chunk[chunk_idx - idx] = image[:, x:x+patch_size[1], y:y+patch_size[2], z:z+patch_size[3]]
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


class TrainingDataSetFixedVolume(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, patch_size=32, step=16, fixed_volume:int=None):
        logging.info(f"Initializing DataSet: data.shape={data.shape}, patch_size={patch_size}, step={step}, fixed_volume={fixed_volume}")
        data = np.transpose(data, (3, 0, 1, 2))
        logging.info(f"Data transposed to: {data.shape}")
        windows = sliding_windows(data, patch_size, step)
        self.windows = torch.from_numpy(windows).type(torch.float)
        logging.info(
            f"Created {self.windows.shape[0]} windows of shape {self.windows.shape[1:]}."
        )
        self.n_volumes = self.windows.shape[1]
        self.fixed_volume = fixed_volume
        self.total_samples = len(self.windows)
        logging.info(f"Total training samples: {self.total_samples:,} (windows: {len(self.windows):,} × volumes: {self.n_volumes})")
        self.take_volumes = [i for i in range(self.n_volumes) if i != self.fixed_volume]
    
    def __getitem__(self, index: int):
        x = self.windows[:, self.take_volumes]
        y = self.windows[:, self.fixed_volume : self.fixed_volume + 1]
        return x, y
    
    def __len__(self):
        """Total number of training samples (windows × volumes)"""
        length = self.total_samples
        logging.debug(f"__len__ called, returning {length}")
        return length
    

class TrainingDataSetMultipleVolumes(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, patch_size=32, step=16):
        logging.info(
            f"Initializing DataSet: data.shape={data.shape}, patch_size={patch_size}, step={step}"
        )
        # transpose data from (X, Y, Z, Bvalues) to (Bvalues, X, Y, Z)
        self.n_vols = data.shape[-1]
        data = np.transpose(data, (3, 0, 1, 2))
        logging.info(f"Data transposed to: {data.shape}")
        windows = sliding_windows(data, patch_size, step)
        self.windows = torch.from_numpy(windows).type(torch.float)
        logging.info(
            f"Created {self.windows.shape[0]} windows of shape {self.windows.shape[1:]}."
        )
        self.n_volumes = self.windows.shape[1]
        
        # Each window will be analyzed n_vols times (once for each target volume)
        self.total_samples = len(self.windows) * self.n_volumes
        logging.info(f"Total training samples: {self.total_samples:,} (windows: {len(self.windows):,} × volumes: {self.n_volumes})")

    def __getitem__(self, index: int):
        # Calculate which window and which volume to use as target
        window_idx = index // self.n_volumes
        target_volume_idx = index % self.n_volumes
        
        # Create input by taking all volumes except the target
        take_volumes = [i for i in range(self.n_volumes) if i != target_volume_idx]
        x = self.windows[window_idx, take_volumes]
        y = self.windows[window_idx, target_volume_idx : target_volume_idx + 1]
        
        # Create volume indices tensor for sinusoidal encoding
        volume_indices = torch.tensor(take_volumes, dtype=torch.long)
        
        logging.debug(
            f"__getitem__ index={index}, window_idx={window_idx}, target_vol={target_volume_idx}, "
            f"x.shape={x.shape}, y.shape={y.shape}, volume_indices={volume_indices.tolist()}"
        )
        return x, y, volume_indices

    def __len__(self):
        """Total number of training samples (windows × volumes)"""
        length = self.total_samples
        logging.debug(f"__len__ called, returning {length}")
        return length


class ReconstructionDataSet(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray):
        logging.info(f"Initializing DataSet: data.shape={data.shape}")
        # transpose data from (X, Y, Z, Bvalues) to (Bvalues, X, Y, Z)
        self.n_vols = data.shape[-1]
        self.data = torch.from_numpy(np.transpose(data, (3, 0, 1, 2))).type(
            torch.float
        )
        logging.info(f"Data transposed to: {self.data.shape}")

    def __getitem__(self, index):
        take_volumes = [i for i in range(self.n_vols) if i != index]
        x = self.data[take_volumes]
        y = self.data[index : index + 1]
        
        # Create volume indices tensor for sinusoidal encoding
        volume_indices = torch.tensor(take_volumes, dtype=torch.long)
        
        logging.debug(
            f"ReconstructionDataSet __getitem__ index={index}, "
            f"x.shape={x.shape}, y.shape={y.shape}, volume_indices={volume_indices.tolist()}"
        )
        return x, y, volume_indices

    def __len__(self):
        """number of volumes"""
        logging.debug(f"__len__ called, returning {self.n_vols}")
        return self.n_vols
