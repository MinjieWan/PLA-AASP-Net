"""Phase-lattice aggregation (PLA)."""

from __future__ import annotations

import math
from typing import Tuple
import torch
from torch import Tensor, nn

from .common import LayerNorm2d, ParallelWKVScan


class PhaseLatticeAggregation(nn.Module):
    """PLA: aggregate unordered support evidence into fixed phase slots, then scan the lattice."""

    def __init__(
        self,
        channels: int,
        scale: int,
        phase_channels: int = 12,
        temperature: float = 0.04,
    ):
        super().__init__()
        if scale < 2:
            raise ValueError("PLA requires scale >= 2")
        if phase_channels < 4:
            raise ValueError("phase_channels must be at least 4")
        if temperature <= 0:
            raise ValueError("phase temperature must be positive")
        self.channels = channels
        self.scale = scale
        self.phase_channels = phase_channels
        self.temperature = temperature
        hidden = max(channels // 2, 8)
        self.score = nn.Sequential(
            nn.Conv2d(3 * channels + 4, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.delta_project = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, phase_channels, 1, bias=False),
        )
        self.phase_norm = LayerNorm2d(phase_channels)
        self.phase_key = nn.Conv2d(
            phase_channels, phase_channels, 1, bias=False
        )
        self.phase_value = nn.Conv2d(
            phase_channels, phase_channels, 1, bias=False
        )
        self.phase_receptance = nn.Conv2d(
            phase_channels, phase_channels, 1, bias=False
        )
        self.phase_scan = ParallelWKVScan(phase_channels)
        self.phase_output = nn.Conv2d(
            phase_channels, phase_channels, 1, bias=False
        )
        self.collapse_project = nn.Conv2d(
            phase_channels, channels, 1, bias=False
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        centers = torch.arange(scale, dtype=torch.float32) / float(scale)
        yy, xx = torch.meshgrid(centers, centers, indexing="ij")
        phase_centers = torch.stack((xx, yy), dim=-1).reshape(1, 1, -1, 2, 1, 1)
        self.register_buffer("phase_centers", phase_centers, persistent=False)

    def _phase_assignment(self, phase: Tensor) -> Tensor:
        # Periodic distance keeps subpixel phases continuous across zero and one.
        difference = (
            phase.float().unsqueeze(2) - self.phase_centers.to(phase.device)
        ).abs()
        difference = torch.minimum(difference, 1.0 - difference)
        logits = -difference.square().sum(dim=3) / self.temperature
        return torch.softmax(logits, dim=2).to(phase.dtype)

    def _phase_wkv(
        self,
        key: Tensor,
        value: Tensor,
        support: Tensor,
    ) -> Tensor:
        batch, phase_count, channels, height, width = key.shape
        if phase_count != self.scale * self.scale:
            raise ValueError("The phase slot count must equal scale squared")
        side = self.scale
        key_grid = key.view(batch, side, side, channels, height, width).permute(
            0, 4, 5, 1, 2, 3
        )
        value_grid = value.view(
            batch, side, side, channels, height, width
        ).permute(0, 4, 5, 1, 2, 3)
        support_grid = support.view(
            batch, side, side, 1, height, width
        ).permute(0, 4, 5, 1, 2, 3)

        row_key = key_grid.reshape(batch * height * width * side, side, channels)
        row_value = value_grid.reshape(
            batch * height * width * side, side, channels
        )
        row_support = support_grid.reshape(
            batch * height * width * side, side, 1
        )
        row_forward = self.phase_scan(row_key, row_value, row_support)
        row_backward = self.phase_scan(
            row_key.flip(1), row_value.flip(1), row_support.flip(1)
        ).flip(1)
        row = 0.5 * (row_forward + row_backward)
        row = row.view(batch, height, width, side, side, channels)

        column_key = key_grid.permute(0, 1, 2, 4, 3, 5).reshape(
            batch * height * width * side, side, channels
        )
        column_value = value_grid.permute(0, 1, 2, 4, 3, 5).reshape(
            batch * height * width * side, side, channels
        )
        column_support = support_grid.permute(0, 1, 2, 4, 3, 5).reshape(
            batch * height * width * side, side, 1
        )
        column_forward = self.phase_scan(
            column_key, column_value, column_support
        )
        column_backward = self.phase_scan(
            column_key.flip(1),
            column_value.flip(1),
            column_support.flip(1),
        ).flip(1)
        column = 0.5 * (column_forward + column_backward)
        column = column.view(
            batch, height, width, side, side, channels
        ).permute(0, 1, 2, 4, 3, 5)
        output = 0.5 * (row + column)
        return output.permute(0, 3, 4, 5, 1, 2).reshape(
            batch, phase_count, channels, height, width
        )

    def forward(
        self,
        features: Tensor,
        confidence: Tensor,
        effective_offsets: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if features.ndim != 5 or confidence.ndim != 5:
            raise ValueError("features/confidence must have shape [B,T,C,H,W]/[B,T,1,H,W]")
        if effective_offsets.ndim != 5 or effective_offsets.size(2) != 2:
            raise ValueError("effective_offsets must have shape [B,T,2,H,W]")
        if (
            features.shape[:2] != confidence.shape[:2]
            or features.shape[:2] != effective_offsets.shape[:2]
            or features.shape[-2:] != confidence.shape[-2:]
            or features.shape[-2:] != effective_offsets.shape[-2:]
        ):
            raise ValueError("Features, confidence, and offsets must have matching frame counts and spatial dimensions")
        batch, frames, channels, height, width = features.shape
        reference = features[:, 0]
        phase_count = self.scale * self.scale
        if frames == 1:
            empty = features.new_zeros(batch, 0, 1, height, width)
            phase_latent = features.new_zeros(
                batch, phase_count, self.phase_channels, height, width
            )
            phase_coverage = features.new_zeros(
                batch, phase_count, 1, height, width
            )
            reliability = features.new_ones(batch, 1, height, width)
            return reference, empty, phase_latent, phase_coverage, reliability

        neighbors = features[:, 1:]
        ref_expand = reference.unsqueeze(1).expand(-1, frames - 1, -1, -1, -1)
        delta = neighbors - ref_expand
        phase = torch.remainder(effective_offsets[:, 1:].float(), 1.0).to(
            features.dtype
        )
        phase_code = torch.cat(
            (
                torch.sin(2.0 * math.pi * phase[:, :, :1]),
                torch.cos(2.0 * math.pi * phase[:, :, :1]),
                torch.sin(2.0 * math.pi * phase[:, :, 1:]),
                torch.cos(2.0 * math.pi * phase[:, :, 1:]),
            ),
            dim=2,
        )
        score_input = torch.cat(
            (neighbors, ref_expand, delta.abs(), phase_code), dim=2
        )
        flat_frames = batch * (frames - 1)
        score = self.score(
            score_input.reshape(flat_frames, 3 * channels + 4, height, width)
        ).view(batch, frames - 1, 1, height, width)
        frame_weight = confidence[:, 1:] * torch.sigmoid(score)
        assignment = self._phase_assignment(phase)
        joint_weight = assignment * frame_weight

        projected = self.delta_project(
            delta.reshape(flat_frames, channels, height, width)
        ).view(batch, frames - 1, self.phase_channels, height, width)
        phase_numerator = torch.einsum(
            "bnphw,bnchw->bpchw", joint_weight, projected
        )
        phase_coverage = joint_weight.sum(dim=1).unsqueeze(2)
        phase_evidence = phase_numerator / (1.0 + phase_coverage)

        phase_flat = phase_evidence.reshape(
            batch * phase_count, self.phase_channels, height, width
        )
        normalized = self.phase_norm(phase_flat)
        key = self.phase_key(normalized).view(
            batch, phase_count, self.phase_channels, height, width
        )
        value = self.phase_value(normalized).view(
            batch, phase_count, self.phase_channels, height, width
        )
        receptance = torch.sigmoid(self.phase_receptance(normalized)).view(
            batch, phase_count, self.phase_channels, height, width
        )
        support = phase_coverage / (1.0 + phase_coverage)
        phase_context = self._phase_wkv(key, value, support)
        phase_context = phase_evidence + self.residual_scale * self.phase_output(
            (receptance * phase_context).reshape(
                batch * phase_count, self.phase_channels, height, width
            )
        ).view(batch, phase_count, self.phase_channels, height, width)

        collapsed = (support * phase_context).sum(dim=1) / (
            support.sum(dim=1) + 1e-6
        )
        fused = reference + self.collapse_project(collapsed)

        # Risk measures written support residuals only; without support evidence reliability is one.
        residual_risk = torch.tanh(delta.abs().mean(dim=2, keepdim=True))
        risk = (frame_weight * residual_risk).sum(dim=1) / (
            1.0 + frame_weight.sum(dim=1)
        )
        reliability = (1.0 - risk).clamp(0.05, 1.0)
        return fused, frame_weight, phase_context, phase_coverage, reliability
