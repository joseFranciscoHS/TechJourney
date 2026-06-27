"""
Export D-Brain (gt, noisy) .npy volumes with the same spatial crop and shell
slice as drcnet_hybrid_rgs (for MP-PCA and fair baseline comparison).

Example:
  python experiments/paper_export_dbrain_volume_pair.py \\
    --config src/drcnet_hybrid_rgs/config.yaml \\
    --out-dir /tmp/paper_pilot_data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Repo src on path when run from repo root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_SRC = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "src"))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

import numpy as np  # noqa: E402

from utils.data import DBrainDataLoader  # noqa: E402
from utils.utils import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=os.path.join(_REPO_SRC, "drcnet_hybrid_rgs", "config.yaml"),
        help="YAML containing dbrain.data / dbrain.model (for RGS G,K)",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--noise-sigma",
        type=float,
        default=None,
        help="Optional override for dbrain.data.noise_sigma before export.",
    )
    args = p.parse_args()

    settings = load_config(args.config).dbrain
    if args.noise_sigma is not None:
        settings.data.noise_sigma = float(args.noise_sigma)
    os.makedirs(args.out_dir, exist_ok=True)

    dl = DBrainDataLoader(
        nii_path=settings.data.nii_path,
        bvecs_path=settings.data.bvecs_path,
        bvalue=settings.data.bvalue,
        noise_sigma=settings.data.noise_sigma,
        noise_type=getattr(settings.data, "noise_type", "rician"),
        n_coils=getattr(settings.data, "noise_n_coils", 1),
    )
    original_data, noisy_data = dl.load_data()

    mode = getattr(settings.data, "shell_sampling_mode", "sequential")
    if mode in {"rgs", "sequential"}:
        take_volumes = settings.data.num_b0s + int(
            getattr(settings.data, "shell_gradient_volumes", settings.data.num_volumes)
        )
    else:
        take_volumes = settings.data.num_b0s + settings.data.num_volumes

    tx, ty, tz = settings.data.take_x, settings.data.take_y, settings.data.take_z
    gt_full = original_data[:tx, :ty, :tz, :take_volumes]
    noisy_crop = noisy_data[:tx, :ty, :tz, settings.data.num_b0s : take_volumes]
    gt_dwi = original_data[:tx, :ty, :tz, settings.data.num_b0s : take_volumes]
    norm_params = getattr(dl, "norm_params_", None)

    np.save(os.path.join(args.out_dir, "gt_full_xyzv.npy"), gt_full.astype(np.float32))
    np.save(os.path.join(args.out_dir, "gt_dwi_xyzv.npy"), gt_dwi.astype(np.float32))
    np.save(
        os.path.join(args.out_dir, "noisy_dwi_xyzv.npy"), noisy_crop.astype(np.float32)
    )
    if norm_params is not None:
        np.save(
            os.path.join(args.out_dir, "norm_params.npy"),
            np.asarray(norm_params[:take_volumes], dtype=np.float64),
        )

    meta = {
        "take_x": int(tx),
        "take_y": int(ty),
        "take_z": int(tz),
        "num_b0s": int(settings.data.num_b0s),
        "take_volumes": int(take_volumes),
        "shell_sampling_mode": mode,
        "noise_sigma": float(settings.data.noise_sigma),
        "bvalue": int(settings.data.bvalue),
        "norm_params_saved": norm_params is not None,
    }
    with open(
        os.path.join(args.out_dir, "export_meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, indent=2)
    logging.info("Wrote npy + export_meta.json to %s", args.out_dir)


if __name__ == "__main__":
    main()
