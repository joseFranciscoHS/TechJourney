import argparse
import importlib.util
import json
from pathlib import Path


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main():
    parser = argparse.ArgumentParser(
        description="Probe runtime dependencies for protocol smoke."
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = {
        "dipy": _has_module("dipy"),
        "wandb": _has_module("wandb"),
        "numpy": _has_module("numpy"),
        "torch": _has_module("torch"),
    }
    payload["status"] = "ok" if payload["dipy"] else "blocked_missing_dipy"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
