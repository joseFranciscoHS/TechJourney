import json
import os
from typing import Dict, Optional

import dipy.reconst.dti as dti
import numpy as np
from dipy.core.gradients import gradient_table


def compute_dti_maps(data_xyzv: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray):
    gtab = gradient_table(bvals, bvecs)
    tenfit = dti.TensorModel(gtab).fit(data_xyzv)
    return {"fa": tenfit.fa, "md": tenfit.md, "ad": tenfit.ad, "rd": tenfit.rd}


def compute_dti_errors(
    denoised_xyzv: np.ndarray,
    gt_xyzv: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    d_dti = compute_dti_maps(denoised_xyzv, bvals, bvecs)
    g_dti = compute_dti_maps(gt_xyzv, bvals, bvecs)
    if roi_mask is None:
        roi_mask = np.ones_like(g_dti["fa"], dtype=bool)

    def mae(a, b):
        return float(np.mean(np.abs(a[roi_mask] - b[roi_mask])))

    return {
        "fa_mae": mae(d_dti["fa"], g_dti["fa"]),
        "md_mae": mae(d_dti["md"], g_dti["md"]),
        "ad_mae": mae(d_dti["ad"], g_dti["ad"]),
        "rd_mae": mae(d_dti["rd"], g_dti["rd"]),
    }


def save_dti_metrics(metrics: Dict[str, float], out_dir: str, name: str = "dti_metrics.json"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
