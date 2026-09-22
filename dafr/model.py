from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


EXPERTS = ("lowbit", "npr", "spectrum")


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = resnet18(weights=None)
        self.network.maxpool = nn.Identity()
        self.network.fc = nn.Identity()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


class Projection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class DualSpaceAdapter(nn.Module):
    def __init__(self, dimension: int = 256, bottleneck: int = 64):
        super().__init__()
        self.authenticity_norm = nn.LayerNorm(dimension)
        self.source_norm = nn.LayerNorm(dimension)
        self.shared_down = nn.Linear(dimension, bottleneck)
        self.activation = nn.GELU()
        self.authenticity_up = nn.Linear(bottleneck, dimension)
        self.source_up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.authenticity_up.weight)
        nn.init.zeros_(self.authenticity_up.bias)
        nn.init.zeros_(self.source_up.weight)
        nn.init.zeros_(self.source_up.bias)

    def forward(
        self, authenticity: torch.Tensor, source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.activation(self.shared_down(self.authenticity_norm(authenticity)))
        s = self.activation(self.shared_down(self.source_norm(source)))
        return (
            F.normalize(authenticity + self.authenticity_up(a), dim=1),
            F.normalize(source + self.source_up(s), dim=1),
        )


class SourceResidualAdapter(nn.Module):
    def __init__(self, dimension: int = 512, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.up(self.activation(self.down(self.norm(feature))))


class DAFR(nn.Module):
    def __init__(
        self,
        source_names: Sequence[str] = (),
        common_dim: int = 512,
        embedding_dim: int = 256,
        adapter_dim: int = 64,
        image_size: int = 256,
    ):
        super().__init__()
        self.source_names = tuple(source_names)
        self.common_dim = common_dim
        self.embedding_dim = embedding_dim
        self.adapter_dim = adapter_dim
        self.image_size = image_size

        self.expert_encoders = nn.ModuleDict(
            {name: ResNet18Encoder() for name in EXPERTS}
        )
        self.expert_projections = nn.ModuleDict(
            {name: Projection(512, common_dim) for name in EXPERTS}
        )
        self.rgb_encoder = ResNet18Encoder()
        self.rgb_projection = Projection(512, 128)
        self.reliability_gate = nn.Sequential(
            nn.Linear(128 + 3 * common_dim, common_dim),
            nn.GELU(),
            nn.Linear(common_dim, 3),
        )
        self.fusion = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
        )
        self.source_residual_adapter = SourceResidualAdapter(common_dim, adapter_dim)
        self.authenticity_projection = nn.Sequential(
            nn.Linear(common_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.source_projection = nn.Sequential(
            nn.Linear(common_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.dual_space_adapter = DualSpaceAdapter(embedding_dim, adapter_dim)
        self.binary_head = nn.Linear(embedding_dim, 2)
        self.source_head = nn.Linear(embedding_dim, max(1, len(self.source_names)))
        self.source_adversary = nn.Linear(embedding_dim, max(1, len(self.source_names)))
        self.expert_binary_heads = nn.ModuleDict(
            {name: nn.Linear(common_dim, 2) for name in EXPERTS}
        )
        self.expert_source_heads = nn.ModuleDict(
            {name: nn.Linear(common_dim, max(1, len(self.source_names))) for name in EXPERTS}
        )

    def configuration(self) -> dict[str, object]:
        return {
            "source_names": list(self.source_names),
            "common_dim": self.common_dim,
            "embedding_dim": self.embedding_dim,
            "adapter_dim": self.adapter_dim,
            "image_size": self.image_size,
        }

    def forward(
        self, views: dict[str, torch.Tensor], grl_weight: float = 1.0,
        include_auxiliary: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        features = {
            name: self.expert_projections[name](self.expert_encoders[name](views[name]))
            for name in EXPERTS
        }
        context = self.rgb_projection(self.rgb_encoder(views["rgb"]))
        weights = self.reliability_gate(torch.cat([context, *(features[name] for name in EXPERTS)], dim=1)).softmax(dim=1)
        stacked = torch.stack([features[name] for name in EXPERTS], dim=1)
        fused = self.fusion((weights.unsqueeze(-1) * stacked).sum(dim=1))
        base_a = F.normalize(self.authenticity_projection(fused), dim=1)
        adapted_fused = self.source_residual_adapter(fused)
        base_s = F.normalize(self.source_projection(adapted_fused), dim=1)
        authenticity, source = self.dual_space_adapter(base_a, base_s)
        result = {
            "expert_features": stacked,
            "expert_weights": weights,
            "fused_feature": fused,
            "authenticity_embedding": authenticity,
            "source_embedding": source,
            "binary_logits": self.binary_head(authenticity),
        }
        if include_auxiliary is None:
            include_auxiliary = self.training
        if include_auxiliary:
            result.update({
                "source_logits": self.source_head(source),
                "adversarial_logits": self.source_adversary(
                    GradientReversal.apply(authenticity, grl_weight)
                ),
                "expert_binary_logits": torch.stack(
                    [self.expert_binary_heads[name](features[name]) for name in EXPERTS], dim=1
                ),
                "expert_source_logits": torch.stack(
                    [self.expert_source_heads[name](features[name]) for name in EXPERTS], dim=1
                ),
            })
        return result
