import argparse
from pathlib import Path
from typing import Optional

import yaml


def _repo_root() -> Path:
    # .../DWMRI/experiments/this_file.py -> repo root
    return Path(__file__).resolve().parent.parent.parent


def _resolve_cwd(cwd: str, repo: Path) -> str:
    p = Path(cwd)
    if p.is_absolute():
        return str(p.resolve())
    return str((repo / p).resolve())


def _parse_k_values(
    k_args: list[int], k_list: Optional[str], parser: argparse.ArgumentParser
) -> list[int]:
    ks: list[int] = list(k_args)
    if k_list:
        for part in k_list.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ks.append(int(part))
            except ValueError:
                parser.error(f"invalid integer in --k-list: {part!r}")
    if not ks:
        parser.error("at least one K is required (use --k and/or --k-list)")
    # Unique K preserves order; duplicate ids would collide in driver.py log paths.
    return list(dict.fromkeys(ks))


def make_job(
    *,
    architecture: str,
    dataset: str,
    regime: str,
    g: int,
    k: int,
    epochs: int,
    sampling_mode: str,
    cwd: str,
    output_root_prefix: Optional[str],
    no_wandb: bool,
    extra_sets: list[str],
) -> dict:
    if k < 1:
        raise ValueError(f"K must be >= 1, got {k}")
    module = (
        "drcnet_hybrid_rgs.run"
        if architecture == "drcnet"
        else "restormer_hybrid_rgs.run"
    )
    job_id = f"{architecture}_3d_k{k}_{dataset}"
    cmd = ["python", "-m", module, "--dataset", dataset, "--regime", regime]
    pfx = f"{dataset}."

    if regime == "supervised":
        cmd += ["--set", f"{pfx}train.supervised=true"]

    cmd += [
        "--set",
        f"{pfx}train.progressive.enabled=false",
        "--set",
        f"{pfx}train.num_epochs={epochs}",
        "--set",
        f"{pfx}data.shell_sampling_mode={sampling_mode}",
        "--set",
        f"{pfx}model.in_channel={k}",
        "--set",
        f"{pfx}data.num_input_volumes={k}",
        "--set",
        f"{pfx}data.target_channel={k - 1}",
        "--set",
        f"{pfx}data.shell_gradient_volumes={g}",
    ]
    for s in extra_sets:
        cmd += ["--set", s]
    if no_wandb:
        cmd += ["--no-wandb"]
    if output_root_prefix is not None:
        out = Path(output_root_prefix).expanduser() / f"K_{k}"
        cmd += ["--output-root", str(out), "--exp-id", job_id]

    return {"id": job_id, "recipe": "k_sweep", "cwd": cwd, "command": cmd}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate K-sweep experiment job manifests for the experiment driver."
    )
    parser.add_argument(
        "--architecture", required=True, choices=["drcnet", "restormer"]
    )
    parser.add_argument("--dataset", default="dbrain")
    parser.add_argument(
        "--g", type=int, required=True, help="data.shell_gradient_volumes (G)"
    )
    parser.add_argument(
        "--k",
        action="append",
        type=int,
        default=argparse.SUPPRESS,
        help="K value (repeatable)",
    )
    parser.add_argument(
        "--k-list",
        default=None,
        metavar="K,K,...",
        help="Comma-separated K values (merged after any --k)",
    )
    parser.add_argument("--regime", default="self_supervised")
    parser.add_argument(
        "--cwd",
        default="DWMRI/src",
        help="Working directory for jobs: absolute path or relative to repo root (default: DWMRI/src)",
    )
    parser.add_argument(
        "--out", required=True, help="Output YAML path for the job manifest"
    )
    parser.add_argument(
        "--output-root-prefix",
        default=None,
        help="If set, each job gets --output-root PREFIX/K_{k} and --exp-id matching the job id",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Append --no-wandb to each command"
    )
    parser.add_argument(
        "--extra-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra --set KEY=VALUE (repeatable)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="train.num_epochs override (default: 5)"
    )
    parser.add_argument(
        "--sampling-mode",
        default="sequential",
        help="data.shell_sampling_mode (default: sequential)",
    )
    args = parser.parse_args()

    repo = _repo_root()
    cwd_resolved = _resolve_cwd(args.cwd, repo)
    k_values = _parse_k_values(list(getattr(args, "k", [])), args.k_list, parser)

    jobs = [
        make_job(
            architecture=args.architecture,
            dataset=args.dataset,
            regime=args.regime,
            g=args.g,
            k=k,
            epochs=args.epochs,
            sampling_mode=args.sampling_mode,
            cwd=cwd_resolved,
            output_root_prefix=args.output_root_prefix,
            no_wandb=args.no_wandb,
            extra_sets=list(args.extra_set),
        )
        for k in k_values
    ]

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"jobs": jobs}, f, sort_keys=False)

    print(
        "Example: python DWMRI/experiments/generate_k_sweep_jobs.py --architecture drcnet --g 10 "
        "--k 12 --k 24 --out DWMRI/experiments/k_sweep_drcnet.yaml"
    )


if __name__ == "__main__":
    main()
