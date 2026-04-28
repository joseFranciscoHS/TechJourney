import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from restormer_hybrid_rgs.model2d import Restormer2D
from utils.data import DBrainDataLoader
from utils.experiment_runtime import append_registry_line, gpu_peak_mem_mb, now_utc_iso


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Sequential2DDataset(Dataset):
    def __init__(self, data_xyzv, k=24, target_channel=23, mask_p=0.3):
        self.data = np.transpose(data_xyzv, (3, 0, 1, 2))  # (V,X,Y,Z)
        self.k = k
        self.tc = target_channel
        self.mask_p = mask_p
        self.windows = self.data.shape[0] - k + 1
        self.slices = self.data.shape[3]

    def __len__(self):
        return self.windows * self.slices

    def __getitem__(self, idx):
        w = idx // self.slices
        z = idx % self.slices
        block = self.data[w : w + self.k, :, :, z]
        x = torch.from_numpy(block).float()
        mask = (torch.rand_like(x[self.tc]) > self.mask_p).float()
        x[self.tc] = x[self.tc] * mask
        y = torch.from_numpy(block[self.tc : self.tc + 1]).float()
        return x, mask.unsqueeze(0), y


def count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--target-channel", type=int, default=23)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--registry-path", default=None)
    p.add_argument("--exp-id", default=None)
    p.add_argument("--job-id", default=None)
    p.add_argument("--recipe", default="restormer_2d")
    args = p.parse_args()
    start = now_utc_iso()
    t0 = time.time()
    device = _auto_device()
    loader = DBrainDataLoader()
    gt, noisy = loader.load_data()
    noisy = noisy[:128, :128, :96, 6:66]
    ds = Sequential2DDataset(noisy, k=args.k, target_channel=args.target_channel)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = Restormer2D(inp_channels=args.k, out_channels=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    for _ in range(args.epochs):
        for x, _m, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            opt.step()

    payload = {
        "schema_version": "v1",
        "exp_id": args.exp_id,
        "job_id": args.job_id,
        "recipe": args.recipe,
        "status": "success",
        "timestamps": {"start_utc": start, "end_utc": now_utc_iso()},
        "duration_s": time.time() - t0,
        "architecture": "restormer",
        "dimensionality": "2d",
        "regime": "self_supervised",
        "control_metrics": {
            "n_params": count_params(model),
            "sec_per_epoch": (time.time() - t0) / max(args.epochs, 1),
            "sec_per_volume": None,
            "peak_gpu_mem_mb": gpu_peak_mem_mb(device),
        },
    }
    append_registry_line(args.registry_path, payload)


if __name__ == "__main__":
    main()
