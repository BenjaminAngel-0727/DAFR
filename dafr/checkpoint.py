from __future__ import annotations

from pathlib import Path

import torch

from .model import DAFR


def save_checkpoint(model: DAFR, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"format": "dafr-paper-resnet18-v1", "config": model.configuration(),
         "state_dict": model.state_dict()},
        destination,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> DAFR:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "dafr-paper-resnet18-v1":
        raise ValueError("Checkpoint is not compatible with the paper-aligned ResNet-18 model")
    model = DAFR(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device)
