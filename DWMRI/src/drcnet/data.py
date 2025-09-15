import logging

import numpy as np
import torch

np.random.seed(91021)


def sliding_windows(image, patch_size, step):
    """
    Memory-efficient chunked sliding window generation.
    """
    logging.info(
        f"Creating sliding windows (memory-efficient): patch_size={patch_size}, step={step}"
    )
    
    # Calculate number of patches per dimension
    patches_per_dim = [(image.shape[i] - patch_size[i]) // step + 1 for i in range(len(patch_size))]
    total_patches = np.prod(patches_per_dim)
    
    logging.info(f"Will generate {total_patches:,} patches with dimensions {patches_per_dim}")
    
    # Process in chunks to avoid memory issues
    chunk_size = min(5000, total_patches)  # Process 5k patches at a time
    result_chunks = []
    
    idx = 0
    while idx < total_patches:
        end_idx = min(idx + chunk_size, total_patches)
        chunk_patches = end_idx - idx
        
        # Pre-allocate chunk
        chunk = np.zeros((chunk_patches, *patch_size), dtype=image.dtype)
        
        # Fill chunk
        chunk_idx = 0
        for v in range(0, image.shape[0] - patch_size[0] + 1, step):
            for x in range(0, image.shape[1] - patch_size[1] + 1, step):
                for y in range(0, image.shape[2] - patch_size[2] + 1, step):
                    for z in range(0, image.shape[3] - patch_size[3] + 1, step):
                        if idx <= chunk_idx < end_idx:
                            chunk[chunk_idx - idx] = image[v:v+patch_size[0], x:x+patch_size[1], y:y+patch_size[2], z:z+patch_size[3]]
                        chunk_idx += 1
                        if chunk_idx >= end_idx:
                            break
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
        
        logging.debug(
            f"__getitem__ index={index}, window_idx={window_idx}, target_vol={target_volume_idx}, "
            f"x.shape={x.shape}, y.shape={y.shape}"
        )
        return x, y

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
        return x, y

    def __len__(self):
        """number of volumes"""
        logging.debug(f"__len__ called, returning {self.n_vols}")
        return self.n_vols
