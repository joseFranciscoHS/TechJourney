"""Export Stanford noisy / P2S / MP-PCA arms into the shared CSD study layout.

All three arms write physical-intensity 4D volumes via
`paper_eval.export_denoised.save_arm` under
`tmp/paper_final_k16_stanford_fixels/arrays/<arm>/` so they match the hybrid
runner exports (correctness rule 5).

Usage (from DWMRI/src, CPU-ok):
  python -m paper_eval.export_stanford_baseline_arms --arms noisy
  python -m paper_eval.export_stanford_baseline_arms --arms noisy,mppca,p2s
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Sequence

import numpy as np
from dipy.denoise.localpca import mppca
from dipy.io.image import load_nifti

from paper_eval.export_denoised import save_arm
from utils.data import StanfordDataLoader, invert_normalization
from utils.utils import load_config

logging.basicConfig(level=logging.INFO)


def _take_volumes(settings) -> int:
    return int(settings.data.num_b0s) + int(
        getattr(settings.data, "shell_gradient_volumes", settings.data.num_volumes)
    )


def _load_stanford_crop(settings):
    """Return (crop_01, norm_params, affine, bvals, bvecs, take_volumes, nb0)."""
    loader = StanfordDataLoader(
        bvalue=settings.data.bvalue, noise_sigma=settings.data.noise_sigma
    )
    original, noisy = loader.load_data()
    if original is None:
        original = noisy
    take_volumes = _take_volumes(settings)
    nb0 = int(settings.data.num_b0s)
    tx, ty, tz = settings.data.take_x, settings.data.take_y, settings.data.take_z
    crop_01 = original[:tx, :ty, :tz, :take_volumes].astype(np.float32)
    norm_params = loader.norm_params_[:take_volumes]
    affine = loader.affine_
    gtab = loader.load_gradient_table()
    bvals = np.asarray(gtab.bvals)[:take_volumes]
    bvecs = np.asarray(gtab.bvecs)[:take_volumes]
    return crop_01, norm_params, affine, bvals, bvecs, take_volumes, nb0


def export_noisy(out_root: str, settings) -> dict:
    crop_01, norm_params, affine, bvals, bvecs, take_volumes, nb0 = _load_stanford_crop(
        settings
    )
    vol_4d = invert_normalization(crop_01.astype(np.float64), norm_params)
    arm_dir = os.path.join(out_root, "noisy")
    paths = save_arm(arm_dir, "noisy", vol_4d.astype(np.float32), affine, bvals, bvecs, nb0)
    # Also persist the normalized crop + norm_params for MP-PCA re-use.
    np.save(os.path.join(arm_dir, "noisy_norm_01.npy"), crop_01)
    np.save(
        os.path.join(arm_dir, "norm_params.npy"),
        np.asarray(norm_params, dtype=np.float64),
    )
    logging.info("Exported noisy arm take_volumes=%d shape=%s", take_volumes, vol_4d.shape)
    return paths


def export_mppca(out_root: str, settings, patch_radius: int = 2) -> dict:
    crop_01, norm_params, affine, bvals, bvecs, take_volumes, nb0 = _load_stanford_crop(
        settings
    )
    # Capture native MP-PCA output BEFORE any eval-protocol rescale (rule 2).
    denoised_native, sigma = mppca(
        crop_01.astype(np.float64), patch_radius=patch_radius, return_sigma=True
    )
    vol_4d = invert_normalization(denoised_native.astype(np.float64), norm_params)
    arm_dir = os.path.join(out_root, "mppca")
    paths = save_arm(
        arm_dir, "mppca", vol_4d.astype(np.float32), affine, bvals, bvecs, nb0
    )
    np.save(os.path.join(arm_dir, "sigma.npy"), sigma)
    logging.info("Exported mppca arm shape=%s", vol_4d.shape)
    return paths


def export_p2s(
    out_root: str,
    settings,
    p2s_nifti: Optional[str] = None,
) -> dict:
    """Re-crop + denorm an existing P2S Stanford NIfTI into the shared layout.

    P2S writes a normalized full-FOV volume; we crop to the training FOV and
    invert the Stanford per-volume min-max so CSD sees physical intensities.
    """
    crop_01, norm_params, affine, bvals, bvecs, take_volumes, nb0 = _load_stanford_crop(
        settings
    )
    if p2s_nifti is None:
        # Default: June/May paper_final Stanford P2S (dipy OLS).
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidates = [
            os.path.join(
                root,
                "tmp/paper_final_k16_out/p2s/output/stanford/"
                "bvalue_2000/noise_sigma_0.01/backend_dipy_model_ols/"
                "denoised_patch2self.nii.gz",
            ),
            os.path.join(
                root,
                "tmp/paper_final_k16_out/p2s/output/stanford/"
                "bvalue_2000/noise_sigma_0.01/denoised_patch2self.nii.gz",
            ),
        ]
        p2s_nifti = next((p for p in candidates if os.path.isfile(p)), None)
        if p2s_nifti is None:
            raise FileNotFoundError(
                "No default P2S Stanford NIfTI found; pass --p2s-nifti explicitly. "
                f"Tried: {candidates}"
            )

    p2s_full, p2s_affine = load_nifti(p2s_nifti)
    tx, ty, tz = settings.data.take_x, settings.data.take_y, settings.data.take_z
    p2s_crop = p2s_full[:tx, :ty, :tz, :take_volumes].astype(np.float64)

    # Prefer the live Stanford loader params; fall back to recompute if shapes
    # disagree (should not happen for the canonical Stanford HARDI).
    if p2s_crop.shape[-1] != take_volumes:
        raise ValueError(
            f"P2S crop has V={p2s_crop.shape[-1]}, expected take_volumes={take_volumes}"
        )
    # Sanity: if the file already looks physical (max >> 1), skip invert.
    if float(np.nanmax(p2s_crop)) > 1.5:
        logging.warning(
            "P2S crop max=%.3f looks physical already; saving without invert",
            float(np.nanmax(p2s_crop)),
        )
        vol_4d = p2s_crop.astype(np.float32)
    else:
        vol_4d = invert_normalization(p2s_crop, norm_params).astype(np.float32)

    # Affine: crop origin is (0,0,0) so the source affine is exact (rule 3).
    # Prefer the live Stanford loader affine; fall back to the P2S file affine.
    use_affine = affine if affine is not None else p2s_affine
    arm_dir = os.path.join(out_root, "p2s")
    paths = save_arm(arm_dir, "p2s", vol_4d, use_affine, bvals, bvecs, nb0)
    logging.info("Exported p2s arm from %s shape=%s", p2s_nifti, vol_4d.shape)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export Stanford noisy/P2S/MP-PCA arms for the CSD fixel study"
    )
    parser.add_argument(
        "--arms",
        default="noisy,mppca,p2s",
        help="Comma-separated subset of: noisy,mppca,p2s",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="arrays/ root (default: tmp/paper_final_k16_stanford_fixels/arrays)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to restormer_hybrid_rgs/config.yaml (for Stanford crop knobs)",
    )
    parser.add_argument("--p2s-nifti", default=None)
    parser.add_argument("--mppca-patch-radius", type=int, default=2)
    args = parser.parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(
        script_dir, "..", "restormer_hybrid_rgs", "config.yaml"
    )
    settings = load_config(config_path).stanford

    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    out_root = args.out_root or os.path.join(
        repo_root, "tmp", "paper_final_k16_stanford_fixels", "arrays"
    )
    os.makedirs(out_root, exist_ok=True)

    arms = [a.strip().lower() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm == "noisy":
            export_noisy(out_root, settings)
        elif arm == "mppca":
            export_mppca(out_root, settings, patch_radius=args.mppca_patch_radius)
        elif arm == "p2s":
            export_p2s(out_root, settings, p2s_nifti=args.p2s_nifti)
        else:
            raise ValueError(f"Unknown arm {arm!r}; expected noisy|mppca|p2s")

    logging.info("Baseline arm export done under %s", out_root)


if __name__ == "__main__":
    main()
