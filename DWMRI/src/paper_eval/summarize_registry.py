import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _flatten(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        out.append(
            {
                "dataset": r.get("dataset"),
                "architecture": r.get("architecture"),
                "status": r.get("status"),
                "n_context_samples": (r.get("inference_config") or {}).get(
                    "n_context_samples"
                ),
                "n_preds": (r.get("inference_config") or {}).get("n_preds"),
                "sec_per_volume": (r.get("control_metrics") or {}).get(
                    "sec_per_volume"
                ),
                "sec_per_epoch": (r.get("control_metrics") or {}).get("sec_per_epoch"),
                "n_params": (r.get("control_metrics") or {}).get("n_params"),
                "peak_gpu_mem_mb": (r.get("control_metrics") or {}).get(
                    "peak_gpu_mem_mb"
                ),
                "psnr": (r.get("quality_metrics_full") or {}).get("psnr"),
                "psnr_roi": (r.get("quality_metrics_roi") or {}).get("psnr"),
                "fa_mae": (r.get("dti_metrics") or {}).get("fa_mae"),
                "md_mae": (r.get("dti_metrics") or {}).get("md_mae"),
            }
        )
    return out


def _write_csv(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Flatten runtime registry JSONL to CSV."
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = _read_jsonl(Path(args.registry))
    rows = _flatten(records)
    _write_csv(rows, Path(args.out))
    print(json.dumps({"rows": len(rows), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
