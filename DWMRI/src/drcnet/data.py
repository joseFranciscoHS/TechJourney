import logging

import numpy as np
import torch
from skimage.util import view_as_windows

np.random.seed(91021)


def sliding_windows(image, patch_size, step):
    logging.info(
        f"Creating sliding windows: patch_size={patch_size}, step={step}"
    )
    image = view_as_windows(image, patch_size, step)
    patches_dims = image.shape[
        : len(patch_size)
    ]  # Number of patches per dimension.
    logging.debug(
        f"Sliding windows shape: {image.shape}, patches_dims: {patches_dims}"
    )
    image = np.reshape(
        image, (np.prod(patches_dims), *patch_size)
    )  # "list" of patches.
    logging.info(f"Sliding windows reshaped to: {image.shape}")
    return image


class DataSet(torch.utils.data.Dataset):
    def __init__(
        self, data: np.ndarray, take_volume_idx=0, patch_size=32, step=16
    ):
        logging.info(
            f"Initializing DataSet: data.shape={data.shape}, take_volume_idx={take_volume_idx}, patch_size={patch_size}, step={step}"
        )
        # transpose data from (X, Y, Z, Bvalues) to (Bvalues, X, Y, Z)
        self.n_vols = data.shape[-1]
        data = np.transpose(data, (3, 0, 1, 2))
        logging.debug(f"Data transposed to: {data.shape}")
        windows = sliding_windows(data, patch_size, step)
        self.windows = torch.from_numpy(windows).type(torch.float)
        logging.info(
            f"Created {self.windows.shape[0]} windows of shape {self.windows.shape[1:]}."
        )
        self.take_volume_idx = take_volume_idx
        self.take_volumes = [
            i for i in range(self.n_vols) if i != take_volume_idx
        ]

    def __getitem__(self, index):
        x = self.windows[index, self.take_volumes]
        y = self.windows[index, self.take_volume_idx]
        logging.debug(
            f"__getitem__ index={index}, x.shape={x.shape}, y.shape={y.shape}"
        )
        return x, y

    def __len__(self):
        """number of volumes"""
        length = len(self.windows)
        logging.debug(f"__len__ called, returning {length}")
        return length
