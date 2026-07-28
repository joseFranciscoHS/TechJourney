#!/usr/bin/env python3
"""Finalize MDS2S D-Brain export into the shared paper array layout.

Converts a DWI-only (normalized) MDS2S reconstruct into physical-intensity
`[b0s + DWIs]` with bvals/bvecs under arrays/mds2s/, matching hybrid exports.

Usage (from DWMRI/src):
  python -m paper_eval.finalize_mds2s_dbrain_volume
  python -m paper_eval.finalize_mds2s_dbrain_volume --src .../denoised_mds2s_raw.npy
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from paper_eval.export_denoised import save_arm
from utils.data import DBrainDataLoader, invert_normalization

logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARRAYS = ROOT / "tmp" / "paper_final_k16_dbrain_exports" / "arrays"
SHARED = ROOT / "tmp" / "paper_final_shared_npy"
NII = "/teamspace/s3_folders/dwmri-dataset/D_BRAIN_b2500_6_60_14_HCP_nless.nii"
BVECS = "/teamspace/s3_folders/dwmri-dataset/D_BRAIN_b2500_6_60_HCP_b_matrix.txt"
NB0, N_DWI, TAKE = 6, 60, 66
TX, TY, TZ = 128, 128, 96


def _to_xyzv_dwi(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 4:
        raise ValueError(f"expected 4D, got {a.shape}")
    if a.shape[-1] == N_DWI:
        return a.astype(np.float64)
    if a.shape[1] == N_DWI:
        # (Z, V, X, Y) -> (X, Y, Z, V)
        return np.transpose(a, (2, 3, 0, 1)).astype(np.float64)
    if a.shape[-1] == TAKE:
        return a[..., NB0:].astype(np.float64)
    raise ValueError(f"unrecognized MDS2S shape {a.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arrays-root",
        type=Path,
        default=DEFAULT_ARRAYS,
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Optional raw/xyzv npy; default searches arrays/mds2s/",
    )
    args = parser.parse_args()
    mdir = args.arrays_root / "mds2s"
    src = args.src
    if src is None:
        for cand in (
            mdir / "denoised_mds2s_raw.npy",
            mdir / "denoised_mds2s.npy",
        ):
            if cand.exists():
                src = cand
                break
    if src is None or not src.exists():
        raise SystemExit(f"no MDS2S source volume under {mdir}")

    dwi_01 = _to_xyzv_dwi(np.load(src))
    # Crop to shared take_z.
    dwi_01 = dwi_01[:TX, :TY, :TZ, :N_DWI]
    gt01 = np.load(SHARED / "gt_full_xyzv.npy")[:TX, :TY, :TZ, :TAKE]
    params = np.load(SHARED / "norm_params.npy")[:TAKE]
    if float(np.nanmax(dwi_01)) > 1.5:
        logging.info("MDS2S already looks physical; skipping DWI denorm")
        dwi_phys = dwi_01.astype(np.float32)
        b0_phys = invert_normalization(gt01[..., :NB0], params[:NB0]).astype(np.float32)
    else:
        b0_phys = invert_normalization(gt01[..., :NB0], params[:NB0]).astype(np.float32)
        dwi_phys = invert_normalization(dwi_01, params[NB0:TAKE]).astype(np.float32)
    vol = np.concatenate([b0_phys, dwi_phys], axis=-1).astype(np.float32)

    loader = DBrainDataLoader(nii_path=NII, bvecs_path=BVECS, bvalue=2500, noise_sigma=0.1)
    gtab = loader.load_gradient_table()
    bvals = np.asarray(gtab.bvals)[:TAKE]
    bvecs = np.asarray(gtab.bvecs)[:TAKE]
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.4
    paths = save_arm(str(mdir), "mds2s", vol, affine, bvals, bvecs, NB0)
    logging.info("Finalized MDS2S arm: %s shape=%s", paths["npy"], vol.shape)


if __name__ == "__main__":
    main()
