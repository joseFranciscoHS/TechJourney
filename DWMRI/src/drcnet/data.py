import numpy as np
import torch
from skimage.util import view_as_windows

np.random.seed(91021)


def sliding_windows(image, patch_size, step):
    image = view_as_windows(image, patch_size, step)
    patches_dims = image.shape[
        : len(patch_size)
    ]  # Number of patches per dimension.
    image = np.reshape(
        image, (np.prod(patches_dims), *patch_size)
    )  # "list" of patches.

    return image


class DataSet(torch.utils.data.Dataset):
    def __init__(
        self, data: np.ndarray, take_volume_idx=0, patch_size=32, step=16
    ):
        # transpose data from (X, Y, Z, Bvalues) to (Bvalues, X, Y, Z)
        self.n_vols = data.shape[-1]
        data = np.transpose(data, (3, 0, 1, 2))
        windows = sliding_windows(data, patch_size, step)
        self.windows = torch.from_numpy(windows).type(torch.float)
        self.take_volume_idx = take_volume_idx
        self.take_volumes = [
            i for i in range(self.n_vols) if i != take_volume_idx
        ]

    def __getitem__(self, index):
        x = self.windows[index, self.take_volumes]
        y = self.windows[index, self.take_volume_idx]
        return x, y

    def __len__(self):
        """number of volumes"""
        return len(self.windows)
