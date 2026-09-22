from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dafr.checkpoint import save_checkpoint
from dafr.data import ManifestDataset, PairedManifestDataset, move_batch, read_manifest
from dafr.forensic_views import ForensicTransform
from dafr.model import DAFR
from dafr.objectives import base_objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = read_manifest(args.train)
    validation_rows = read_manifest(args.validation)
    sources = sorted({row["source"] for row in train_rows if row["binary_label"] == "1"})
    model = DAFR(source_names=sources, image_size=args.image_size).to(args.device)
    model.source_residual_adapter.requires_grad_(False)
    transform = ForensicTransform(size=args.image_size)
    train_data = PairedManifestDataset(train_rows, sources, transform=transform)
    validation_data = ManifestDataset(validation_rows, sources, transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, num_workers=0)
    encoder_parameters = list(model.expert_encoders.parameters()) + list(model.rgb_encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    other_parameters = [parameter for parameter in model.parameters()
                        if id(parameter) not in encoder_ids and parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_parameters, "lr": 3e-5},
         {"params": other_parameters, "lr": 1e-4}], weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    total_steps = max(1, args.epochs * len(train_loader))
    step = 0
    best_score = -1.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            clean_views, degraded_views, binary, source = batch
            device = torch.device(args.device)
            clean_views = {name: tensor.to(device) for name, tensor in clean_views.items()}
            degraded_views = {name: tensor.to(device) for name, tensor in degraded_views.items()}
            binary = binary.to(device)
            source = source.to(device)
            reversal = min(1.0, step / max(1, int(total_steps * 0.3)))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=args.device.startswith("cuda")):
                output = model(clean_views, grl_weight=reversal)
                degraded_output = model(degraded_views, grl_weight=0.0,
                                        include_auxiliary=False)
                losses = base_objective(output, binary, source)
                detection_consistency = F.mse_loss(
                    output["binary_logits"].softmax(1),
                    degraded_output["binary_logits"].softmax(1),
                )
                source_consistency = (
                    1.0 - F.cosine_similarity(
                        output["source_embedding"],
                        degraded_output["source_embedding"], dim=1
                    )
                ).mean()
                total_loss = losses["total"] + 0.2 * detection_consistency + 0.1 * source_consistency
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(total_loss.detach())
            step += 1

        model.eval()
        correct_binary = 0
        correct_source = 0
        count_binary = 0
        count_source = 0
        with torch.inference_mode():
            for batch in validation_loader:
                views, binary, source = move_batch(batch, torch.device(args.device))
                output = model(views, grl_weight=0.0, include_auxiliary=True)
                correct_binary += int((output["binary_logits"].argmax(1) == binary).sum())
                count_binary += len(binary)
                known = (binary == 1) & (source >= 0)
                if known.any():
                    correct_source += int((output["source_logits"][known].argmax(1) == source[known]).sum())
                    count_source += int(known.sum())
        accuracy = correct_binary / max(1, count_binary)
        source_accuracy = correct_source / max(1, count_source)
        score = accuracy + source_accuracy if count_source else accuracy
        print(f"epoch={epoch + 1} loss={train_loss / len(train_loader):.4f} "
              f"binary_acc={accuracy:.4f} source_acc={source_accuracy:.4f}")
        if score > best_score:
            best_score = score
            save_checkpoint(model, args.checkpoint)


if __name__ == "__main__":
    main()
