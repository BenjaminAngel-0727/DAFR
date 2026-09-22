from __future__ import annotations

import argparse
import json

import torch
from PIL import Image

from dafr.checkpoint import load_checkpoint
from dafr.registry import SourceRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    registry = SourceRegistry.load(args.registry)
    registry.refresh(model, device)
    with Image.open(args.image) as image:
        result = registry.predict(model, image, device, top_k=args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
