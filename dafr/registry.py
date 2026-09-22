from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from .forensic_views import ForensicTransform
from .model import DAFR


class SourceRegistry:
    def __init__(self, references: dict[str, list[str]] | None = None):
        self.references = references or {}
        self.embeddings: torch.Tensor | None = None
        self.labels: list[str] = []

    @classmethod
    def load(cls, path: str | Path) -> "SourceRegistry":
        contents = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({str(k): list(v) for k, v in contents["references"].items()})

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"references": self.references}, indent=2), encoding="utf-8"
        )

    def register(self, source: str, image_paths: list[str | Path]) -> None:
        if not source:
            raise ValueError("source must not be empty")
        paths = [str(Path(path).resolve()) for path in image_paths]
        if not paths:
            raise ValueError("At least one reference image is required")
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        current = self.references.setdefault(source, [])
        current.extend(path for path in paths if path not in current)
        self.embeddings = None

    @torch.inference_mode()
    def refresh(
        self, model: DAFR, device: torch.device,
        transform: ForensicTransform | None = None,
    ) -> None:
        if not self.references:
            raise ValueError("Cannot refresh an empty registry")
        model.eval()
        transform = transform or ForensicTransform(size=model.image_size)
        vectors = []
        labels = []
        for source, paths in self.references.items():
            for path in paths:
                with Image.open(path) as image:
                    views = transform(image)
                batch = {name: tensor.unsqueeze(0).to(device) for name, tensor in views.items()}
                vector = model(batch, grl_weight=0.0)["source_embedding"].cpu()
                vectors.append(vector)
                labels.append(source)
        self.embeddings = F.normalize(torch.cat(vectors, dim=0), dim=1)
        self.labels = labels

    def rank(self, query_embedding: torch.Tensor, top_k: int = 10) -> list[dict[str, object]]:
        if self.embeddings is None:
            raise RuntimeError("Call refresh after every model update or registry change")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if query_embedding.shape[0] != 1:
            raise ValueError("rank expects one query embedding")
        query = F.normalize(query_embedding.detach().cpu(), dim=1)
        scores = (query @ self.embeddings.T).squeeze(0)
        indices = scores.argsort(descending=True)[:top_k]
        return [
            {"source": self.labels[index], "similarity": float(scores[index])}
            for index in indices.tolist()
        ]

    @torch.inference_mode()
    def predict(
        self, model: DAFR, image: Image.Image, device: torch.device,
        transform: ForensicTransform | None = None, top_k: int = 10,
    ) -> dict[str, object]:
        model.eval()
        transform = transform or ForensicTransform(size=model.image_size)
        views = transform(image)
        batch = {name: tensor.unsqueeze(0).to(device) for name, tensor in views.items()}
        output = model(batch, grl_weight=0.0)
        probabilities = output["binary_logits"].softmax(dim=1)[0]
        if probabilities.argmax().item() == 0:
            return {"authenticity": "real", "fake_probability": float(probabilities[1]),
                    "source": None, "ranking": []}
        ranking = self.rank(output["source_embedding"], top_k)
        return {"authenticity": "fake", "fake_probability": float(probabilities[1]),
                "source": ranking[0]["source"] if ranking else None, "ranking": ranking}
