import argparse
import json
import os
import subprocess
import time

import yaml


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_job(job, exp_id, output_root, registry_path, fail_fast=False):
    cmd = list(job["command"])
    cmd += ["--exp-id", exp_id, "--job-id", job["id"], "--recipe", job.get("recipe", job["id"])]
    cmd += ["--output-root", output_root, "--registry-path", registry_path]
    log_path = os.path.join(output_root, "runs", f"{job['id']}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=job.get("cwd"), stdout=logf, stderr=logf)
    if proc.returncode != 0 and fail_fast:
        raise RuntimeError(f"Job failed: {job['id']} (exit {proc.returncode})")
    return {"id": job["id"], "exit_code": proc.returncode, "duration_s": time.time() - t0}


def main():
    parser = argparse.ArgumentParser(description="Sequential experiment driver")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    summary = []
    for job in manifest.get("jobs", []):
        summary.append(
            run_job(
                job,
                exp_id=args.exp_id,
                output_root=args.output_root,
                registry_path=args.registry_path,
                fail_fast=args.fail_fast,
            )
        )
    with open(os.path.join(args.output_root, "driver_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
