from __future__ import annotations

import torch
from torch.nn import functional as F


def decorrelation_loss(authenticity: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    if authenticity.shape[0] < 2:
        return authenticity.sum() * 0.0
    centered_a = authenticity - authenticity.mean(dim=0, keepdim=True)
    centered_s = source - source.mean(dim=0, keepdim=True)
    covariance = centered_a.T @ centered_s / (authenticity.shape[0] - 1)
    return covariance.square().mean()


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    vectors = F.normalize(embeddings.float(), dim=1)
    similarity = vectors @ vectors.T / temperature
    eye = torch.eye(len(vectors), dtype=torch.bool, device=vectors.device)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    valid = positives.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0
    log_probability = similarity - torch.logsumexp(
        similarity.masked_fill(eye, -torch.inf), dim=1, keepdim=True
    )
    mean_positive = (log_probability.masked_fill(~positives, 0.0).sum(dim=1)
                     / positives.sum(dim=1).clamp_min(1))
    return -mean_positive[valid].mean()


def base_objective(
    output: dict[str, torch.Tensor],
    binary_targets: torch.Tensor,
    source_targets: torch.Tensor,
    reliability_temperature: float = 0.5,
    source_error_weight: float = 1.0,
    contrastive_temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    synthetic = (binary_targets == 1) & (source_targets >= 0)
    binary = F.cross_entropy(output["binary_logits"], binary_targets)
    expert_binary_errors = torch.stack(
        [F.cross_entropy(output["expert_binary_logits"][:, k], binary_targets, reduction="none")
         for k in range(3)], dim=1
    )
    expert_binary = expert_binary_errors.mean()
    zero = binary * 0.0
    source = zero
    adversarial = zero
    contrastive = zero
    expert_source = zero
    expert_source_errors = torch.zeros_like(expert_binary_errors)
    if synthetic.any():
        labels = source_targets[synthetic]
        source = F.cross_entropy(output["source_logits"][synthetic], labels)
        adversarial = F.cross_entropy(output["adversarial_logits"][synthetic], labels)
        contrastive = supervised_contrastive_loss(
            output["source_embedding"][synthetic], labels, contrastive_temperature
        )
        errors = torch.stack(
            [F.cross_entropy(output["expert_source_logits"][synthetic, k], labels, reduction="none")
             for k in range(3)], dim=1
        )
        expert_source_errors[synthetic] = errors
        expert_source = errors.mean()
    joint_error = expert_binary_errors + source_error_weight * expert_source_errors
    target = (-joint_error.detach() / reliability_temperature).softmax(dim=1)
    reliability = F.kl_div(output["expert_weights"].clamp_min(1e-8).log(), target, reduction="batchmean")
    decorrelation = decorrelation_loss(
        output["authenticity_embedding"], output["source_embedding"]
    )
    total = (binary + source + 0.1 * contrastive + 0.1 * adversarial
             + 0.2 * reliability + 0.05 * decorrelation
             + 0.1 * expert_binary + 0.1 * expert_source)
    return {
        "total": total,
        "binary": binary,
        "source": source,
        "contrastive": contrastive,
        "adversarial": adversarial,
        "reliability": reliability,
        "decorrelation": decorrelation,
        "expert_binary": expert_binary,
        "expert_source": expert_source,
    }
