import argparse

import yaml


def make_job(architecture: str, dimensionality: str, regime: str, dataset: str):
    module = (
        "drcnet_hybrid_rgs.run"
        if architecture == "drcnet"
        else "restormer_hybrid_rgs.run"
    )
    cmd = ["python", "-m", module, "--dataset", dataset, "--regime", regime]
    if dimensionality == "2d":
        cmd += ["--set", "model.use_2d=true"]
    if regime == "supervised":
        cmd += ["--set", "train.supervised=true"]
    job_id = f"{architecture}_{dimensionality}_{regime}_{dataset}"
    return {
        "id": job_id,
        "recipe": "model_dim_matrix",
        "cwd": "DWMRI/src",
        "command": cmd,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.matrix, "r", encoding="utf-8") as f:
        m = yaml.safe_load(f)
    jobs = []
    for d in m["matrix"]["datasets"]:
        for r in m["matrix"]["regimes"]:
            for mdl in m["matrix"]["models"]:
                jobs.append(make_job(mdl["architecture"], mdl["dimensionality"], r, d))
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump({"jobs": jobs}, f, sort_keys=False)


if __name__ == "__main__":
    main()
