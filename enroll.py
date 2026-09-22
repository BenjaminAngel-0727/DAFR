from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader

from dafr.adaptation import adaptation_objective, configure_adaptation
from dafr.checkpoint import load_checkpoint, save_checkpoint
from dafr.data import ManifestDataset, move_batch, read_manifest
from dafr.forensic_views import ForensicTransform
from dafr.registry import SourceRegistry


def repeat(loader):
    while True:
        yield from loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--anchors")
    parser.add_argument("--registry")
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-registry", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    references = read_manifest(args.references)
    if any(row["binary_label"] != "1" for row in references):
        raise ValueError("The references manifest must contain generated images only")
    anchors = read_manifest(args.anchors) if args.anchors else []
    if args.steps > 0 and not any(row["binary_label"] == "0" for row in anchors):
        raise ValueError("Adaptation requires authentic anchors; provide --anchors")
    registry = SourceRegistry.load(args.registry) if args.registry else SourceRegistry()
    retained = [
        {"path": path, "binary_label": "1", "source": source}
        for source, paths in registry.references.items() for path in paths
    ]
    adaptation_rows = references + retained + anchors
    sources = sorted({row["source"] for row in adaptation_rows if row["binary_label"] == "1"})

    if args.steps > 0:
        groups = configure_adaptation(model)
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        dataset = ManifestDataset(adaptation_rows, sources,
                                  transform=ForensicTransform(size=model.image_size))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        batches = repeat(loader)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        for step in range(args.steps):
            warmup = min(1.0, (step + 1) / 20)
            decay = 0.5 * (1.0 + math.cos(math.pi * step / max(1, args.steps)))
            for group, baseline in zip(optimizer.param_groups, (5e-5, 5e-6, 5e-6, 5e-6)):
                group["lr"] = baseline * warmup * decay
            views, binary, source = move_batch(next(batches), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                output = model(views, grl_weight=0.0)
                losses = adaptation_objective(output, binary, source)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if (step + 1) % 100 == 0 or step + 1 == args.steps:
                print(f"step={step + 1} loss={float(losses['total']):.4f}")

    for source in sorted({row["source"] for row in references}):
        registry.register(source, [row["path"] for row in references if row["source"] == source])
    registry.refresh(model, device)
    save_checkpoint(model, args.output_checkpoint)
    registry.save(args.output_registry)
    print(f"registered_sources={len(registry.references)} references={len(registry.labels)}")


if __name__ == "__main__":
    main()
