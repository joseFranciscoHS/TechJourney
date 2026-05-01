from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _table_main(metrics_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    wanted = {
        "mppca",
        "patch2self",
        "mds2s",
        "drcnet_hybrid_rgs",
        "restormer_hybrid_rgs",
    }
    out: List[Dict[str, str]] = []
    for row in metrics_rows:
        arch = (row.get("architecture") or "").strip()
        if arch not in wanted:
            continue
        out.append(
            {
                "method": arch,
                "backend": row.get("backend", ""),
                "dataset": row.get("dataset", ""),
                "psnr": row.get("psnr", ""),
                "ssim": row.get("ssim", ""),
                "mse": row.get("mse", ""),
                "psnr_roi": row.get("psnr_roi", ""),
                "fa_mae": row.get("fa_mae", ""),
                "md_mae": row.get("md_mae", ""),
                "metrics_dir": row.get("metrics_dir", ""),
            }
        )
    return out


def _table_ablations(metrics_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in metrics_rows:
        metrics_dir = row.get("metrics_dir", "")
        if not any(
            tag in metrics_dir for tag in ["sequential", "rgs", "2d_", "_K5", "_K10"]
        ):
            continue
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "architecture": row.get("architecture", ""),
                "backend": row.get("backend", ""),
                "psnr_roi": row.get("psnr_roi", ""),
                "fa_mae": row.get("fa_mae", ""),
                "metrics_dir": metrics_dir,
            }
        )
    return out


def _table_runtime(registry_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in registry_rows:
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "architecture": row.get("architecture", ""),
                "status": row.get("status", ""),
                "n_params": row.get("n_params", ""),
                "sec_per_epoch": row.get("sec_per_epoch", ""),
                "sec_per_volume": row.get("sec_per_volume", ""),
                "peak_gpu_mem_mb": row.get("peak_gpu_mem_mb", ""),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Create dry-run paper tables from pilot CSV outputs."
    )
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    metrics_rows = _read_csv(Path(args.metrics_csv))
    registry_rows = _read_csv(Path(args.registry_csv))
    out_dir = Path(args.out_dir)

    main_rows = _table_main(metrics_rows)
    ablation_rows = _table_ablations(metrics_rows)
    runtime_rows = _table_runtime(registry_rows)

    _write_csv(out_dir / "table_main_dryrun.csv", main_rows)
    _write_csv(out_dir / "table_ablations_dryrun.csv", ablation_rows)
    _write_csv(out_dir / "table_runtime_dryrun.csv", runtime_rows)

    print(
        {
            "status": "ok",
            "table_main_rows": len(main_rows),
            "table_ablations_rows": len(ablation_rows),
            "table_runtime_rows": len(runtime_rows),
            "out_dir": str(out_dir),
        }
    )


if __name__ == "__main__":
    main()
