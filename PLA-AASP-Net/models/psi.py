"""Phase-spatial interaction (PSI) and shared local units."""

from __future__ import annotations

from typing import List
import torch
from torch import Tensor, nn

from .aasp import AlignmentAwareStatePropagation
from .common import LayerNorm2d
from .pla import PhaseLatticeAggregation


class SharedLocalUnit(nn.Module):
    """Shared local unit (SLU) applied independently to each frame in the unordered set."""

    def __init__(self, channels: int, bottleneck: int):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.reduce = nn.Conv2d(channels, bottleneck, 1)
        self.conv3 = nn.Conv2d(
            bottleneck, bottleneck, 3, 1, 1, groups=bottleneck
        )
        self.conv_h = nn.Conv2d(
            bottleneck, bottleneck, (1, 3), 1, (0, 1), groups=bottleneck
        )
        self.conv_v = nn.Conv2d(
            bottleneck, bottleneck, (3, 1), 1, (1, 0), groups=bottleneck
        )
        self.expand = nn.Conv2d(bottleneck, channels, 1)
        self.act = nn.GELU()
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError("SharedLocalUnit input must have shape [B,T,C,H,W]")
        batch, frames, channels, height, width = x.shape
        flat = x.reshape(batch * frames, channels, height, width)
        local = self.reduce(self.norm(flat))
        local = (self.conv3(local) + self.conv_h(local) + self.conv_v(local)) / 3.0
        local = self.expand(self.act(local))
        return x + self.residual_scale * local.view(
            batch, frames, channels, height, width
        )


class PhaseSpatialInteraction(nn.Module):
    """Phase-spatial interaction (PSI) with shared local units, PLA, and AASP."""

    def __init__(
        self,
        channels: int = 48,
        bottleneck: int = 24,
        scale: int = 4,
        phase_channels: int = 12,
        phase_temperature: float = 0.04,
    ):
        super().__init__()
        self.units = nn.ModuleList([SharedLocalUnit(channels, bottleneck) for _ in range(4)])
        self.pools = nn.ModuleList(
            [
                PhaseLatticeAggregation(
                    channels, scale, phase_channels, phase_temperature
                )
                for _ in range(3)
            ]
        )
        self.spatial_rwkv = nn.ModuleList(
            [AlignmentAwareStatePropagation(channels) for _ in range(3)]
        )
        self.inject = nn.ModuleList(
            [nn.Conv2d(channels, channels, 1) for _ in range(3)]
        )
        self.global_fuse = nn.Conv2d(3 * channels, channels, 1)
        self.global_inject = nn.Conv2d(channels, channels, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        features: Tensor,
        confidence: Tensor,
        effective_offsets: Tensor,
    ) -> Tensor:
        original = features
        contexts: List[Tensor] = []
        for index in range(3):
            features = self.units[index](features)
            pooled, _, _, phase_coverage, reliability = self.pools[index](
                features, confidence, effective_offsets
            )
            coverage = 1.0 - torch.exp(-phase_coverage.sum(dim=1))
            context = self.spatial_rwkv[index](pooled, reliability, coverage)
            contexts.append(context)
            injection = self.inject[index](context).unsqueeze(1)
            features = features + confidence * injection
        global_context = self.global_fuse(torch.cat(contexts, dim=1))
        features = features + confidence * self.global_inject(
            global_context
        ).unsqueeze(1)
        features = self.units[3](features)
        return original + self.residual_scale * (features - original)
