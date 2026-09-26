"""Reproducible runtime and checkpoint utilities shared by CLI, training and UI."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = (
            "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS is not available in this Python environment.")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available in this Python environment.")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temp, path)


def environment_info(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
    }


def save_checkpoint(path, model, optimizer=None, *, epoch=0, step=0, training_config=None, metadata=None):
    payload = {
        "format_version": 1,
        "model_config": asdict(model.config),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "epoch": epoch,
        "step": step,
        "training_config": training_config or {},
        "metadata": metadata or {},
        "torch_rng_state": torch.get_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if any(parameter.device.type == "cuda" for parameter in model.parameters()):
        payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def load_checkpoint(path, device="cpu"):
    from .models import ModelConfig, build_model

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format; expected photograder format_version=1.")
    cfg = dict(payload["model_config"])
    cfg["styles"] = tuple(cfg.get("styles", ("Expert C",)))
    model = build_model(ModelConfig(**cfg)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload
