import json
import os
import socket
import time
from typing import Any, Dict, Iterable, Optional

import torch


def parse_override_value(raw: str) -> Any:
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def apply_overrides(settings, overrides: Iterable[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key, raw_value = item.split("=", 1)
        value = parse_override_value(raw_value)
        node = settings
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node:
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value


def apply_output_root(settings, output_root: Optional[str]) -> None:
    if not output_root:
        return
    train_dir = settings.train.checkpoint_dir
    metrics_dir = settings.reconstruct.metrics_dir
    images_dir = settings.reconstruct.images_dir
    settings.train.checkpoint_dir = os.path.join(output_root, train_dir)
    settings.reconstruct.metrics_dir = os.path.join(output_root, metrics_dir)
    settings.reconstruct.images_dir = os.path.join(output_root, images_dir)


def losses_dir_from_train_checkpoint_dir(train_checkpoint_dir: str) -> str:
    """Training-losses root parallel to ``settings.train.checkpoint_dir``.

    Config paths look like ``.../<pkg>/checkpoints/<dataset>``; losses must
    live under ``.../<pkg>/losses/<dataset>`` with the same ``output_root``
    prefix as checkpoints.
    """
    if "checkpoints" in train_checkpoint_dir:
        return train_checkpoint_dir.replace("checkpoints", "losses", 1)
    return train_checkpoint_dir


def gpu_peak_mem_mb(device: str) -> float:
    if not (isinstance(device, str) and device.startswith("cuda")):
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    idx = 0
    if ":" in device:
        idx = int(device.split(":")[1])
    return float(torch.cuda.max_memory_allocated(idx) / (1024 * 1024))


def hardware_info(device: str) -> Dict[str, Any]:
    info = {
        "host": socket.gethostname(),
        "gpu_name": None,
        "gpu_count": 0,
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        idx = 0
        if ":" in device:
            idx = int(device.split(":")[1])
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def append_registry_line(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
