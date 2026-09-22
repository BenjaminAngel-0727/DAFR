from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import DAFR
from .objectives import decorrelation_loss, supervised_contrastive_loss


ADAPTATION_MODULES = (
    "source_residual_adapter",
    "authenticity_projection",
    "source_projection",
    "dual_space_adapter",
)


def configure_adaptation(model: DAFR) -> list[dict[str, object]]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups = []
    for name in ADAPTATION_MODULES:
        module: nn.Module = getattr(model, name)
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        groups.append({
            "params": list(module.parameters()),
            "lr": 5e-5 if name == "source_residual_adapter" else 5e-6,
        })
    return groups


def prototype_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    classes, inverse = labels.unique(sorted=True, return_inverse=True)
    if len(classes) < 2:
        return embeddings.sum() * 0.0
    vectors = F.normalize(embeddings.float(), dim=1)
    counts = torch.bincount(inverse, minlength=len(classes))
    valid = counts[inverse] > 1
    if not valid.any():
        return embeddings.sum() * 0.0
    sums = torch.stack([vectors[inverse == index].sum(dim=0) for index in range(len(classes))])
    prototypes = torch.stack([
        F.normalize(sums[index], dim=0)
        for index in range(len(classes))
    ])
    logits = vectors @ prototypes.T / temperature
    leave_one_out = F.normalize(sums[inverse] - vectors, dim=1)
    own = (vectors * leave_one_out).sum(dim=1)
    logits = logits.scatter(1, inverse[:, None], own[:, None] / temperature)
    return F.cross_entropy(logits[valid], inverse[valid])


def adaptation_objective(
    output: dict[str, torch.Tensor],
    binary_targets: torch.Tensor,
    source_targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    binary = F.cross_entropy(output["binary_logits"], binary_targets)
    synthetic = (binary_targets == 1) & (source_targets >= 0)
    zero = binary * 0.0
    prototype = zero
    contrastive = zero
    if synthetic.any():
        embeddings = output["source_embedding"][synthetic]
        labels = source_targets[synthetic]
        prototype = prototype_loss(embeddings, labels)
        contrastive = supervised_contrastive_loss(embeddings, labels)
    decorrelation = decorrelation_loss(
        output["authenticity_embedding"], output["source_embedding"]
    )
    total = binary + 0.3 * prototype + 0.2 * contrastive + 0.05 * decorrelation
    return {
        "total": total,
        "binary": binary,
        "prototype": prototype,
        "contrastive": contrastive,
        "decorrelation": decorrelation,
    }
