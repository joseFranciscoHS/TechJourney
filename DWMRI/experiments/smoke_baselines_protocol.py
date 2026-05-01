import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from p2s.run import denoise_dwi_patch2self, denoise_dwi_sklearn_reference
from paper_eval.baselines.mppca_run import run_mppca
from paper_eval.consolidate_results import _collect_rows, _write_csv
from paper_eval.dti_metrics import save_dti_metrics, try_compute_dti_errors
from utils.eval_protocol import compute_roi_mask, metrics_policy_dict, save_run_manifest
from utils.metrics import compute_metrics, save_metrics


def _synthetic_pair(seed: int = 7):
    rng = np.random.default_rng(seed)
    x, y, z = 12, 12, 8
    nb0, ndwi = 2, 8
    v = nb0 + ndwi
    base = rng.uniform(0.15, 1.0, size=(x, y, z, 1))
    angles = np.linspace(0, np.pi, ndwi, endpoint=False)
    dwi_clean = np.concatenate(
        [base * (0.6 + 0.4 * np.cos(a) ** 2) for a in angles], axis=-1
    )
    b0_clean = np.concatenate([base * 1.1, base * 1.05], axis=-1)
    gt = np.concatenate([b0_clean, dwi_clean], axis=-1).astype(np.float64)
    noisy = np.clip(gt + rng.normal(0, 0.03, size=gt.shape), 0.0, None)

    bvals = np.array([0.0, 0.0] + [1000.0] * ndwi, dtype=np.float64)
    bvecs = np.zeros((v, 3), dtype=np.float64)
    for i, a in enumerate(angles, start=nb0):
        bvecs[i] = np.array([np.cos(a), np.sin(a), 0.0])
    return gt, noisy, bvals, bvecs


def run(out_root: Path):
    gt, noisy, bvals, bvecs = _synthetic_pair(seed=91021)
    out_root.mkdir(parents=True, exist_ok=True)

    mppca_out = out_root / "mppca" / "dbrain"
    run_mppca(
        noisy_xyzv=noisy,
        gt_xyzv=gt,
        out_dir=str(mppca_out),
        patch_radius=2,
        bvecs_path=None,
    )
    mppca_denoised = np.load(mppca_out / "denoised.npy")
    save_dti_metrics(
        try_compute_dti_errors(mppca_denoised, gt, bvals, bvecs, roi_threshold=0.02),
        str(mppca_out),
    )

    roi_thr = 0.02
    roi_mask = compute_roi_mask(gt, roi_thr)
    policy = metrics_policy_dict(
        reference_name="clean_gt",
        rescale_to_01=True,
        rescale_mode="per_volume",
        clip_to_range=True,
        roi_threshold=roi_thr,
    )

    def _save_p2s(backend: str, den: np.ndarray):
        out = out_root / "p2s" / "dbrain" / f"backend_{backend}"
        out.mkdir(parents=True, exist_ok=True)
        save_metrics(compute_metrics(gt, den), str(out), filename="metrics.json")
        save_metrics(
            compute_metrics(gt, den, mask=roi_mask),
            str(out),
            filename="metrics_roi.json",
        )
        save_dti_metrics(
            try_compute_dti_errors(den, gt, bvals, bvecs, roi_threshold=roi_thr),
            str(out),
        )
        save_run_manifest(
            out_dir=str(out),
            seed=91021,
            reproducible=True,
            runtime_device="cpu",
            config={
                "dataset": "dbrain",
                "architecture": "patch2self",
                "backend": backend,
            },
            metrics_policy=policy,
        )
        return out

    den_dipy = denoise_dwi_patch2self(
        noisy.astype(np.float32),
        bvals,
        SimpleNamespace(
            model="ols", shift_intensity=True, clip_negative_vals=False, b0_threshold=50
        ),
    ).astype(np.float64)
    p2s_dipy_out = _save_p2s("dipy", den_dipy)

    den_sk = denoise_dwi_sklearn_reference(
        noisy.astype(np.float32),
        bvals,
        SimpleNamespace(
            b0_threshold=50,
            sklearn_model="ols",
            patch_radius=[0, 0, 0],
            patch_stride=1,
            use_b0_as_predictors=True,
        ),
    ).astype(np.float64)
    p2s_sk_out = _save_p2s("sklearn_reference", den_sk)

    rows = _collect_rows(out_root)
    csv_out = out_root / "baseline_summary.csv"
    _write_csv(rows, csv_out)
    return {
        "status": "ok",
        "rows": len(rows),
        "csv": str(csv_out),
        "outputs": [str(mppca_out), str(p2s_dipy_out), str(p2s_sk_out)],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Smoke-run baseline protocol with synthetic data."
    )
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    result = run(Path(args.out_root))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
