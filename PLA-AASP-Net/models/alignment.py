"""Pyramid encoding, optical flow, and deformable alignment."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import DeformConv2d
from utils import strip_prefix, torch_load_compat

from .common import ResidualStack


def flow_warp(x: Tensor, flow: Tensor, padding_mode: str = "border") -> Tensor:
    """Sample source features on the reference grid using (x, y) displacements."""
    if x.ndim != 4 or flow.ndim != 4 or flow.size(1) != 2:
        raise ValueError("x must have shape [B,C,H,W] and flow must have shape [B,2,H,W]")
    batch, _, height, width = x.shape
    if flow.shape[-2:] != (height, width):
        raise ValueError("flow and x must have matching spatial dimensions")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=x.device, dtype=x.dtype),
        torch.arange(width, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), -1).unsqueeze(0).expand(batch, -1, -1, -1)
    grid = grid + flow.permute(0, 2, 3, 1).to(x.dtype)
    grid_x = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    return F.grid_sample(
        x,
        torch.stack((grid_x, grid_y), -1),
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


def resize_flow(flow: Tensor, size: Tuple[int, int]) -> Tensor:
    height, width = flow.shape[-2:]
    new_height, new_width = size
    output = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    output[:, 0] *= new_width / max(width, 1)
    output[:, 1] *= new_height / max(height, 1)
    return output


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(8, 32, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 7, 1, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.basic_module(x)


class SPyNet(nn.Module):
    """BasicVSR/EDVR-style SPyNet that samples support frames on the reference grid."""

    def __init__(self, pretrained: Optional[str] = None):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )
        if pretrained is not None:
            missing, unexpected = self.load_pretrained(pretrained)
            if missing or unexpected:
                raise RuntimeError(
                    f"Incompatible SPyNet pretrained weights: missing={missing}, unexpected={unexpected}"
                )

    def load_pretrained(self, path: str) -> Tuple[List[str], List[str]]:
        obj = torch_load_compat(path, "cpu")
        if isinstance(obj, dict):
            for key in ("params", "state_dict", "model", "generator"):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
                    break
        if not isinstance(obj, dict):
            raise TypeError("No state_dict found in the SPyNet checkpoint")
        state = strip_prefix(obj, ("module.", "spynet.", "generator.", "net_g."))
        remapped: Dict[str, Tensor] = {}
        pattern = re.compile(r"^(basic_module\.\d+\.basic_module\.)(\d+)\.conv\.(weight|bias)$")
        for key, value in state.items():
            match = pattern.match(key)
            if match:
                key = f"{match.group(1)}{2 * int(match.group(2))}.{match.group(3)}"
            remapped[key] = value
        message = self.load_state_dict(remapped, strict=False)
        return list(message.missing_keys), list(message.unexpected_keys)

    def _compute_flow(self, ref: Tensor, supp: Tensor) -> Tensor:
        ref_pyramid = [(ref - self.mean) / self.std]
        supp_pyramid = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref_pyramid.insert(
                0, F.avg_pool2d(ref_pyramid[0], 2, 2, count_include_pad=False)
            )
            supp_pyramid.insert(
                0, F.avg_pool2d(supp_pyramid[0], 2, 2, count_include_pad=False)
            )
        batch, _, height, width = ref_pyramid[0].shape
        flow = ref.new_zeros(batch, 2, max(height // 2, 1), max(width // 2, 1))
        for level in range(6):
            flow = F.interpolate(
                flow, size=ref_pyramid[level].shape[-2:], mode="bilinear", align_corners=True
            ) * 2.0
            warped = flow_warp(supp_pyramid[level], flow)
            flow = flow + self.basic_module[level](
                torch.cat((ref_pyramid[level], warped, flow), 1)
            )
        return flow

    def forward(self, ref: Tensor, supp: Tensor) -> Tensor:
        if ref.shape != supp.shape or ref.size(1) != 3:
            raise ValueError("ref and supp must both have shape [B,3,H,W]")
        height, width = ref.shape[-2:]
        resized_height = math.ceil(height / 32) * 32
        resized_width = math.ceil(width / 32) * 32
        ref_resized = F.interpolate(
            ref, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )
        supp_resized = F.interpolate(
            supp, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )
        flow = self._compute_flow(ref_resized, supp_resized)
        flow = F.interpolate(flow, size=(height, width), mode="bilinear", align_corners=True)
        flow[:, 0] *= width / resized_width
        flow[:, 1] *= height / resized_height
        return flow


class PyramidFeatureExtractor(nn.Module):
    def __init__(self, channels: int = 48, blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualStack(channels, blocks),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualStack(channels, 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualStack(channels, 2),
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        level0 = self.stem(x)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        return [level0, level1, level2]


class FlowGuidedDCNLevel(nn.Module):
    def __init__(
        self,
        channels: int,
        deform_groups: int = 8,
        max_residue: float = 10.0,
        with_coarse: bool = False,
    ):
        super().__init__()
        if channels % deform_groups:
            raise ValueError("channels must be divisible by deform_groups")
        self.deform_groups = deform_groups
        self.max_residue = max_residue
        self.with_coarse = with_coarse
        condition_channels = 3 * channels + 2 + (channels if with_coarse else 0)
        self.offset_head = nn.Sequential(
            nn.Conv2d(condition_channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 27 * deform_groups, 3, 1, 1),
        )
        self.dcn = DeformConv2d(channels, channels, 3, 1, 1, bias=True)
        self.fuse = nn.Conv2d(2 * channels, channels, 3, 1, 1) if with_coarse else nn.Identity()
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

    def forward(
        self,
        neighbor: Tensor,
        reference: Tensor,
        flow: Tensor,
        coarse: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        warped = flow_warp(neighbor, flow)
        condition = [reference, warped, (reference - warped).abs(), flow]
        if self.with_coarse:
            if coarse is None:
                raise ValueError("This alignment level requires coarse features")
            coarse = F.interpolate(
                coarse, size=reference.shape[-2:], mode="bilinear", align_corners=False
            )
            condition.append(coarse)
        raw = self.offset_head(torch.cat(condition, 1))
        offset_x, offset_y, mask = torch.chunk(raw, 3, 1)
        residual = self.max_residue * torch.tanh(torch.cat((offset_x, offset_y), 1))
        # Average the DCN sampling residuals to obtain a local subpixel phase correction.
        # This correction constructs the phase lattice; full DCN sampling offsets remain intact.
        residual_mean = torch.stack(
            (offset_x.tanh().mean(1), offset_y.tanh().mean(1)), dim=1
        ) * self.max_residue
        base = flow.flip(1).repeat(1, self.deform_groups * 9, 1, 1)
        mask = torch.sigmoid(mask)
        output = self.dcn(neighbor, base + residual, mask)
        if self.with_coarse:
            output = self.fuse(torch.cat((output, coarse), 1))
        return output, mask.mean(1, keepdim=True), residual_mean


class MultiScaleFlowGuidedAlign(nn.Module):
    def __init__(self, channels: int = 48, deform_groups: int = 8, max_residue: float = 10.0):
        super().__init__()
        self.level2 = FlowGuidedDCNLevel(channels, deform_groups, max_residue, False)
        self.level1 = FlowGuidedDCNLevel(channels, deform_groups, max_residue, True)
        self.level0 = FlowGuidedDCNLevel(channels, deform_groups, max_residue, True)
        hidden = max(channels // 2, 8)
        self.confidence = nn.Sequential(
            nn.Conv2d(3 * channels + 3, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, hidden, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 1, 3, 1, 1),
        )

    def forward(
        self,
        neighbor_pyramid: Sequence[Tensor],
        reference_pyramid: Sequence[Tensor],
        flow: Tensor,
        neighbor_image: Tensor,
        reference_image: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        flow2 = resize_flow(flow, reference_pyramid[2].shape[-2:])
        aligned2, _, _ = self.level2(neighbor_pyramid[2], reference_pyramid[2], flow2)
        flow1 = resize_flow(flow, reference_pyramid[1].shape[-2:])
        aligned1, _, _ = self.level1(
            neighbor_pyramid[1], reference_pyramid[1], flow1, aligned2
        )
        flow0 = resize_flow(flow, reference_pyramid[0].shape[-2:])
        aligned0, modulation, residual_mean = self.level0(
            neighbor_pyramid[0], reference_pyramid[0], flow0, aligned1
        )
        warped_image = flow_warp(neighbor_image, flow)
        photometric = (reference_image - warped_image).abs().mean(1, keepdim=True)
        valid = flow_warp(
            torch.ones_like(neighbor_image[:, :1]), flow, padding_mode="zeros"
        ).clamp(0, 1)
        confidence = torch.sigmoid(
            self.confidence(
                torch.cat(
                    (
                        reference_pyramid[0],
                        aligned0,
                        (reference_pyramid[0] - aligned0).abs(),
                        modulation,
                        photometric,
                        valid,
                    ),
                    1,
                )
            )
        ) * valid
        effective_offset = flow0 + residual_mean
        return aligned0, confidence, effective_offset
