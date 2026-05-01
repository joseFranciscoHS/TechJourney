from __future__ import annotations

import argparse
import os

from paper_eval.baselines.mppca_run import run_mppca
from utils.data import DBrainDataLoader


def main():
    parser = argparse.ArgumentParser(
        description="Run MP-PCA baseline directly on D-Brain loader."
    )
    parser.add_argument("--output-root", default="tmp/paper_pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--nii-path", default=None)
    parser.add_argument("--bvecs-path", default=None)
    args = parser.parse_args()

    loader = DBrainDataLoader(nii_path=args.nii_path, bvecs_path=args.bvecs_path)
    gt, noisy = loader.load_data()
    if gt is None:
        raise RuntimeError(
            "DBrainDataLoader returned no clean GT; MP-PCA baseline requires GT."
        )

    out_dir = os.path.join(
        args.output_root,
        "mppca",
        "bvalue_2500",
        "noise_sigma_0.1",
        "backend_dipy_localpca",
    )
    run_mppca(
        noisy_xyzv=noisy,
        gt_xyzv=gt,
        out_dir=out_dir,
        patch_radius=args.patch_radius,
        bvecs_path=getattr(loader, "bvecs_path", None),
        bvalue=2500.0,
        metrics_roi_threshold=0.02,
        rescale_to_01=True,
        rescale_mode="per_volume",
        clip_to_range=True,
    )
    print(
        {
            "status": "ok",
            "baseline": "mppca",
            "out_dir": out_dir,
            "seed_hint": args.seed,
        }
    )


if __name__ == "__main__":
    main()
