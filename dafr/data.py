from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .forensic_views import ForensicTransform, random_degradation


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest = Path(path).resolve()
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "binary_label", "source"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires path,binary_label,source columns")
        rows = []
        for row in reader:
            image_path = Path(row["path"])
            if not image_path.is_absolute():
                image_path = manifest.parent / image_path
            rows.append({
                "path": str(image_path.resolve()),
                "binary_label": row["binary_label"].strip(),
                "source": row["source"].strip(),
            })
    if not rows:
        raise ValueError("Manifest is empty")
    for row in rows:
        if row["binary_label"] not in {"0", "1"}:
            raise ValueError("binary_label must be 0 or 1")
        if row["binary_label"] == "1" and not row["source"]:
            raise ValueError("Every synthetic image needs a source label")
    return rows


class ManifestDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        source_names: list[str] | tuple[str, ...],
        transform: ForensicTransform | None = None,
    ):
        self.rows = rows
        self.source_index = {name: index for index, name in enumerate(source_names)}
        self.transform = transform or ForensicTransform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            views = self.transform(image)
        binary = int(row["binary_label"])
        source = self.source_index.get(row["source"], -1) if binary else -1
        return views, torch.tensor(binary), torch.tensor(source)


class PairedManifestDataset(ManifestDataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        source_names: list[str] | tuple[str, ...],
        degradation_probability: float = 0.5,
        transform: ForensicTransform | None = None,
    ):
        super().__init__(rows, source_names, transform)
        self.degradation_probability = degradation_probability

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        clean = self.transform(image)
        degraded_image = random_degradation(image) if random.random() < self.degradation_probability else image
        degraded = self.transform(degraded_image)
        binary = int(row["binary_label"])
        source = self.source_index.get(row["source"], -1) if binary else -1
        return clean, degraded, torch.tensor(binary), torch.tensor(source)


def move_batch(batch, device: torch.device):
    views, binary, source = batch
    return (
        {name: tensor.to(device) for name, tensor in views.items()},
        binary.to(device),
        source.to(device),
    )
