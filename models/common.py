"""Shared layers, WKV scan, and upsampling."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class ResidualStack(nn.Module):
    def __init__(self, channels: int, blocks: int):
        super().__init__()
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ParallelWKVScan(nn.Module):
    """Parallel FP32 WKV scan with position-wise evidence and state propagation gates."""

    def __init__(self, channels: int):
        super().__init__()
        initial_decay = torch.linspace(0.70, 0.995, channels).clamp(1e-4, 1 - 1e-4)
        self.decay_logit = nn.Parameter(torch.logit(initial_decay))
        self.time_first = nn.Parameter(torch.full((channels,), math.log(0.5)))

    @staticmethod
    def _prepare_gate(gate: Optional[Tensor], target: Tensor) -> Tensor:
        if gate is None:
            return torch.ones_like(target)
        if gate.ndim == 2:
            gate = gate.unsqueeze(-1)
        if gate.ndim != 3 or gate.shape[:2] != target.shape[:2]:
            raise ValueError("WKV gates must have shape [B,N], [B,N,1], or [B,N,C]")
        if gate.size(-1) not in (1, target.size(-1)):
            raise ValueError("WKV gates must have one channel or match the feature channels")
        return gate.float().expand_as(target).clamp(0.0, 1.0)

    @staticmethod
    def _prefix_scan(
        value: Tensor, weight: Tensor, coefficient: Tensor
    ) -> Tuple[Tensor, Tensor]:
        numerator = weight * value
        denominator = weight
        offset = 1
        while offset < value.size(1):
            old_coefficient = coefficient
            old_numerator = numerator
            old_denominator = denominator
            right_coefficient = old_coefficient[:, offset:]
            coefficient = torch.cat(
                (
                    old_coefficient[:, :offset],
                    right_coefficient * old_coefficient[:, :-offset],
                ),
                dim=1,
            )
            numerator = torch.cat(
                (
                    old_numerator[:, :offset],
                    old_numerator[:, offset:]
                    + right_coefficient * old_numerator[:, :-offset],
                ),
                dim=1,
            )
            denominator = torch.cat(
                (
                    old_denominator[:, :offset],
                    old_denominator[:, offset:]
                    + right_coefficient * old_denominator[:, :-offset],
                ),
                dim=1,
            )
            offset *= 2
        return numerator, denominator

    def forward(
        self,
        key: Tensor,
        value: Tensor,
        source_gate: Optional[Tensor] = None,
        transition_gate: Optional[Tensor] = None,
    ) -> Tensor:
        if key.ndim != 3 or key.shape != value.shape:
            raise ValueError("key and value must both have shape [B,N,C]")
        key32 = key.float()
        value32 = value.float()
        key32 = key32 - key32.amax(dim=1, keepdim=True).detach()
        source = self._prepare_gate(source_gate, key32)
        weight = torch.exp(key32.clamp(min=-30.0, max=0.0)) * source
        decay = torch.sigmoid(self.decay_logit.float()).view(1, 1, -1)
        transition = self._prepare_gate(transition_gate, key32)
        coefficient = decay.expand_as(key32) * transition
        prefix_num, prefix_den = self._prefix_scan(value32, weight, coefficient)
        zeros_num = torch.zeros_like(prefix_num[:, :1])
        zeros_den = torch.zeros_like(prefix_den[:, :1])
        past_num = coefficient * torch.cat((zeros_num, prefix_num[:, :-1]), dim=1)
        past_den = coefficient * torch.cat((zeros_den, prefix_den[:, :-1]), dim=1)
        current_weight = torch.exp(self.time_first.float()).view(1, 1, -1) * weight
        output = (past_num + current_weight * value32) / (
            past_den + current_weight + 1e-6
        )
        return output.to(value.dtype)


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, channels: int, scale: int):
        super().__init__()
        layers: List[nn.Module] = []
        if scale in (2, 4, 8):
            for _ in range(int(math.log2(scale))):
                layers.extend(
                    [
                        nn.Conv2d(channels, 4 * channels, 3, 1, 1),
                        nn.PixelShuffle(2),
                        nn.LeakyReLU(0.1, inplace=True),
                    ]
                )
        elif scale == 3:
            layers.extend(
                [
                    nn.Conv2d(channels, 9 * channels, 3, 1, 1),
                    nn.PixelShuffle(3),
                    nn.LeakyReLU(0.1, inplace=True),
                ]
            )
        else:
            raise ValueError("Supported scales are 2, 3, 4, and 8")
        self.body = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)
