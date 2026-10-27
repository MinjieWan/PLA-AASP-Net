"""Alignment-aware state propagation (AASP)."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import LayerNorm2d, ParallelWKVScan


class MultiScaleSpatialMix(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv3 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.conv5 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels)
        self.scale_logits = nn.Parameter(torch.zeros(3))

    def forward(self, x: Tensor) -> Tensor:
        weights = torch.softmax(self.scale_logits, dim=0)
        return (
            weights[0] * self.conv1(x)
            + weights[1] * self.conv3(x)
            + weights[2] * self.conv5(x)
        )


class AlignmentAwareStatePropagation(nn.Module):
    """AASP: reliability gates evidence writing; reliability and edges gate state retention."""

    def __init__(self, channels: int, expand: int = 2):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.multi_scale = MultiScaleSpatialMix(channels)
        self.receptance = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.scan = ParallelWKVScan(channels)
        self.output = nn.Conv2d(channels, channels, 1)
        gate_hidden = max(channels // 4, 8)
        self.reliability_gate = nn.Sequential(
            nn.Conv2d(3, gate_hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, 2, 1),
        )
        nn.init.zeros_(self.reliability_gate[-1].weight)
        nn.init.zeros_(self.reliability_gate[-1].bias)
        self.norm2 = LayerNorm2d(channels)
        hidden = channels * expand
        self.channel_gate = nn.Conv2d(channels, channels, 1)
        self.channel_key = nn.Conv2d(channels, hidden, 1)
        self.channel_value = nn.Conv2d(hidden, channels, 1)

    @staticmethod
    def _edge_map(x: Tensor) -> Tensor:
        horizontal = (x[..., 1:] - x[..., :-1]).abs().mean(1, keepdim=True)
        vertical = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(1, keepdim=True)
        horizontal = F.pad(horizontal, (0, 1, 0, 0), mode="replicate")
        vertical = F.pad(vertical, (0, 0, 0, 1), mode="replicate")
        edge = horizontal + vertical
        normalizer = edge.mean(dim=(-2, -1), keepdim=True).detach().clamp_min(1e-6)
        return torch.tanh(edge / normalizer)

    def _spatial_wkv(
        self,
        key: Tensor,
        value: Tensor,
        source_gate: Tensor,
        transition_gate: Tensor,
    ) -> Tensor:
        batch, channels, height, width = key.shape
        key_hw = key.permute(0, 2, 3, 1).contiguous()
        value_hw = value.permute(0, 2, 3, 1).contiguous()
        source_hw = source_gate.permute(0, 2, 3, 1).contiguous()
        transition_hw = transition_gate.permute(0, 2, 3, 1).contiguous()

        row_key = key_hw.view(batch * height, width, channels)
        row_value = value_hw.view(batch * height, width, channels)
        row_source = source_hw.view(batch * height, width, 1)
        row_transition = transition_hw.view(batch * height, width, 1)
        row_forward = self.scan(row_key, row_value, row_source, row_transition)
        row_backward = self.scan(
            row_key.flip(1),
            row_value.flip(1),
            row_source.flip(1),
            row_transition.flip(1),
        ).flip(1)
        row = 0.5 * (row_forward + row_backward)
        row = row.view(batch, height, width, channels)

        column_key = key_hw.permute(0, 2, 1, 3).reshape(
            batch * width, height, channels
        )
        column_value = value_hw.permute(0, 2, 1, 3).reshape(
            batch * width, height, channels
        )
        column_source = source_hw.permute(0, 2, 1, 3).reshape(
            batch * width, height, 1
        )
        column_transition = transition_hw.permute(0, 2, 1, 3).reshape(
            batch * width, height, 1
        )
        column_forward = self.scan(
            column_key, column_value, column_source, column_transition
        )
        column_backward = self.scan(
            column_key.flip(1),
            column_value.flip(1),
            column_source.flip(1),
            column_transition.flip(1),
        ).flip(1)
        column = 0.5 * (column_forward + column_backward)
        column = column.view(batch, width, height, channels).permute(0, 2, 1, 3)
        return (0.5 * (row + column)).permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        x: Tensor,
        reliability: Optional[Tensor] = None,
        coverage: Optional[Tensor] = None,
    ) -> Tensor:
        batch, _, height, width = x.shape
        if reliability is None:
            reliability = x.new_ones(batch, 1, height, width)
        if coverage is None:
            coverage = x.new_zeros(batch, 1, height, width)
        if reliability.shape != (batch, 1, height, width):
            raise ValueError("reliability must have shape [B,1,H,W]")
        if coverage.shape != (batch, 1, height, width):
            raise ValueError("coverage must have shape [B,1,H,W]")
        reliability = reliability.clamp(0.0, 1.0)
        coverage = coverage.clamp(0.0, 1.0)

        mixed = self.multi_scale(self.norm1(x))
        edge = self._edge_map(mixed)
        learned = self.reliability_gate(
            torch.cat((reliability, coverage, edge), dim=1)
        )
        # With no support frames, reliability is one and spatial modeling remains active.
        source_gate = (0.05 + 0.95 * reliability) * torch.sigmoid(
            learned[:, :1] + 2.0
        )
        # Limit state propagation at unreliable regions and strong boundaries.
        transition_gate = torch.sigmoid(
            learned[:, 1:] + 3.0 - 2.0 * edge - 2.0 * (1.0 - reliability)
        )
        receptance = torch.sigmoid(self.receptance(mixed))
        context = self._spatial_wkv(
            self.key(mixed),
            self.value(mixed),
            source_gate,
            transition_gate,
        )
        x = x + self.output(receptance * context)
        normalized = self.norm2(x)
        channel = self.channel_value(F.relu(self.channel_key(normalized)).square())
        return x + torch.sigmoid(self.channel_gate(normalized)) * channel
