"""
Merge per-method metrics directories into a single pilot_summary.json.

Each input directory should contain metrics.json (required), and optionally
metrics_roi.json and dti_metrics.json.

Example:
  python experiments/merge_pilot_metrics.py \\
    --inputs drcnet:out/d/metrics p2s:out/p/metrics mppca:out/m \\
    --output pilot_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Pairs name=path/to/metrics_dir (e.g. drcnet:/tmp/m)",
    )
    p.add_argument("--output", required=True, help="Output pilot_summary.json path")
    args = p.parse_args()

    rows: Dict[str, Any] = {}
    for item in args.inputs:
        if "=" not in item:
            raise SystemExit(f"Invalid --inputs entry {item!r}, expected name=path")
        name, path = item.split("=", 1)
        name = name.strip()
        path = os.path.expanduser(path.strip())
        full = _load_json(os.path.join(path, "metrics.json"))
        roi = _load_json(os.path.join(path, "metrics_roi.json"))
        dti = _load_json(os.path.join(path, "dti_metrics.json"))
        rows[name] = {
            "metrics_dir": os.path.abspath(path),
            "metrics": full,
            "metrics_roi": roi,
            "dti_metrics": dti,
        }

    out = {"methods": rows}
    out_abs = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
