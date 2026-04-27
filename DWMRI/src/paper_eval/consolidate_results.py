import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics_dir = metrics_path.parent
        metrics = _read_json(metrics_path) or {}
        metrics_roi = _read_json(metrics_dir / "metrics_roi.json") or {}
        dti = _read_json(metrics_dir / "dti_metrics.json") or {}
        manifest = _read_json(metrics_dir / "run_manifest.json") or {}
        row = {
            "metrics_dir": str(metrics_dir),
            "dataset": (manifest.get("config", {}) or {}).get("dataset"),
            "architecture": (manifest.get("config", {}) or {}).get("architecture"),
            "backend": (manifest.get("config", {}) or {}).get("backend"),
            "seed": manifest.get("seed"),
            "reproducible": manifest.get("reproducible"),
            "psnr": metrics.get("psnr"),
            "ssim": metrics.get("ssim"),
            "mse": metrics.get("mse"),
            "psnr_roi": metrics_roi.get("psnr"),
            "ssim_roi": metrics_roi.get("ssim"),
            "mse_roi": metrics_roi.get("mse"),
            "fa_mae": dti.get("fa_mae"),
            "md_mae": dti.get("md_mae"),
            "ad_mae": dti.get("ad_mae"),
            "rd_mae": dti.get("rd_mae"),
            "dti_reference": dti.get("dti_reference"),
            "dti_skipped_reason": dti.get("dti_skipped_reason"),
            "n_context_samples": (manifest.get("config", {}) or {}).get("n_context_samples"),
            "n_preds": (manifest.get("config", {}) or {}).get("n_preds"),
            "metrics_roi_threshold": (manifest.get("metrics_policy", {}) or {}).get(
                "metrics_roi_threshold"
            ),
            "rescale_to_01": (manifest.get("metrics_policy", {}) or {}).get("rescale_to_01"),
            "rescale_mode": (manifest.get("metrics_policy", {}) or {}).get("rescale_mode"),
            "clip_to_range": (manifest.get("metrics_policy", {}) or {}).get("clip_to_range"),
        }
        rows.append(row)
    return rows


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Consolidate metrics into a single CSV table.")
    parser.add_argument("--root", required=True, help="Root directory to scan recursively.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()
    rows = _collect_rows(Path(args.root))
    _write_csv(rows, Path(args.out))
    print(json.dumps({"rows": len(rows), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
