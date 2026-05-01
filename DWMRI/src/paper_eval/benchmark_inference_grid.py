import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from typing import Dict, List


@dataclass(frozen=True)
class GridPoint:
    n_context: int
    n_preds: int


def _python_module_for_arch(arch: str) -> str:
    if arch == "drcnet":
        return "drcnet_hybrid_rgs.run"
    if arch == "restormer":
        return "restormer_hybrid_rgs.run"
    raise ValueError(f"Unsupported architecture: {arch}")


def run_grid(
    *,
    architecture: str,
    dataset: str,
    checkpoint: str,
    output_root: str,
    registry_path: str,
    n_context_values: List[int],
    n_preds_values: List[int],
    config_path: str | None,
    extra_overrides: List[str],
) -> List[Dict[str, int]]:
    module = _python_module_for_arch(architecture)
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    runs: List[Dict[str, int]] = []
    for n_context, n_preds in product(n_context_values, n_preds_values):
        point = GridPoint(n_context=n_context, n_preds=n_preds)
        cmd = [
            sys.executable,
            "-m",
            module,
            "--dataset",
            dataset,
            "--skip-train",
            "--no-images",
            "--no-wandb",
            "--checkpoint",
            checkpoint,
            "--output-root",
            output_root,
            "--registry-path",
            registry_path,
            "--set",
            f"reconstruct.n_context_samples={point.n_context}",
            "--set",
            f"reconstruct.n_preds={point.n_preds}",
        ]
        if config_path:
            cmd.extend(["--config", config_path])
        for override in extra_overrides:
            cmd.extend(["--set", override])
        subprocess.run(cmd, check=True)
        runs.append({"n_context": point.n_context, "n_preds": point.n_preds})
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference grid for RGS models."
    )
    parser.add_argument(
        "--architecture", choices=["drcnet", "restormer"], required=True
    )
    parser.add_argument("--dataset", choices=["dbrain", "stanford"], default="dbrain")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-context", nargs="+", type=int, default=[12, 24, 48])
    parser.add_argument("--n-preds", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Extra overrides like reconstruct.mask_p=0.3",
    )
    args = parser.parse_args()
    runs = run_grid(
        architecture=args.architecture,
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        registry_path=args.registry_path,
        n_context_values=args.n_context,
        n_preds_values=args.n_preds,
        config_path=args.config,
        extra_overrides=args.overrides,
    )
    print(json.dumps({"completed_runs": runs}, indent=2))


if __name__ == "__main__":
    main()
