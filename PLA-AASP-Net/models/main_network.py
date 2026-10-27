"""PLA-AASP Net: model assembly and forward pass."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .aasp import AlignmentAwareStatePropagation
from .alignment import MultiScaleFlowGuidedAlign, PyramidFeatureExtractor, SPyNet
from .common import PixelShuffleUpsampler, ResidualStack
from .pla import PhaseLatticeAggregation
from .psi import PhaseSpatialInteraction


class PLAAASPNet(nn.Module):
    """PLA-AASP Net for unordered RGB multi-frame super-resolution."""

    def __init__(
        self,
        c: int = 48,
        scale: int = 4,
        feat_blocks: int = 4,
        tssrm_blocks: int = 1,
        tssrm_bottleneck: int = 24,
        recon_blocks: int = 4,
        deform_groups: int = 8,
        max_residue: float = 10.0,
        spynet_pretrained: Optional[str] = None,
        max_flow: float = 32.0,
        phase_channels: int = 12,
        phase_temperature: float = 0.04,
    ):
        super().__init__()
        if tssrm_blocks < 1:
            raise ValueError("tssrm_blocks must be at least 1")
        self.scale = scale
        self.max_flow = max_flow
        self.phase_channels = phase_channels
        self.spynet = SPyNet(spynet_pretrained)
        self.feature = PyramidFeatureExtractor(c, feat_blocks)
        self.align = MultiScaleFlowGuidedAlign(c, deform_groups, max_residue)
        self.condition = nn.Sequential(
            nn.Conv2d(3 * c + 1, c, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, c, 3, 1, 1),
        )
        self.tssrm = nn.ModuleList(
            [
                PhaseSpatialInteraction(
                    c,
                    tssrm_bottleneck,
                    scale,
                    phase_channels,
                    phase_temperature,
                )
                for _ in range(tssrm_blocks)
            ]
        )
        self.final_fusion = PhaseLatticeAggregation(
            c, scale, phase_channels, phase_temperature
        )
        self.final_spatial = AlignmentAwareStatePropagation(c)
        self.local_reference = ResidualStack(c, 3)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(2 * c, c, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualStack(c, recon_blocks),
        )
        self.upsampler = PixelShuffleUpsampler(c, scale)
        self.phase_expand = nn.Conv2d(phase_channels, c, 1, bias=False)
        self.phase_shuffle = nn.PixelShuffle(scale)
        self.phase_refine = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
        )
        self.phase_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.hr = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, 3, 3, 1, 1),
        )
        # Initialize near bicubic while preserving first-step gradients into the backbone.
        nn.init.normal_(self.hr[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.hr[-1].bias)

    def freeze_spynet(self) -> None:
        for parameter in self.spynet.parameters():
            parameter.requires_grad = False
        self.spynet.eval()

    def unfreeze_spynet(self) -> None:
        for parameter in self.spynet.parameters():
            parameter.requires_grad = True

    def parameter_groups(self, base_lr: float, flow_lr_ratio: float = 0.01) -> List[Dict]:
        flow_ids = {id(parameter) for parameter in self.spynet.parameters()}
        main = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in flow_ids and parameter.requires_grad
        ]
        flow = [parameter for parameter in self.spynet.parameters() if parameter.requires_grad]
        groups: List[Dict] = [{"params": main, "lr": base_lr}]
        if flow:
            groups.append({"params": flow, "lr": base_lr * flow_lr_ratio})
        return groups

    @staticmethod
    def _reference_first(x: Tensor, ref_idx: int) -> Tuple[Tensor, List[int]]:
        frames = x.size(1)
        if not 0 <= ref_idx < frames:
            raise ValueError(f"ref_idx must be in [0,{frames - 1}], got {ref_idx}")
        order = [ref_idx] + [index for index in range(frames) if index != ref_idx]
        return x[:, order], order

    def _flows_to_reference(self, x: Tensor) -> List[Tensor]:
        batch, frames, _, height, width = x.shape
        flows = [x.new_zeros(batch, 2, height, width) for _ in range(frames)]
        if frames == 1:
            return flows
        reference = x[:, 0].unsqueeze(1).expand(-1, frames - 1, -1, -1, -1)
        reference = reference.reshape(-1, 3, height, width)
        neighbors = x[:, 1:].reshape(-1, 3, height, width)
        with torch.cuda.amp.autocast(enabled=False):
            batch_flow = self.spynet(reference.float(), neighbors.float())
        batch_flow = batch_flow.clamp(-self.max_flow, self.max_flow)
        batch_flow = batch_flow.view(batch, frames - 1, 2, height, width)
        for index in range(1, frames):
            flows[index] = batch_flow[:, index - 1]
        return flows

    def forward(
        self,
        x: Tensor,
        return_aux: bool = False,
        ref_idx: int = 0,
    ):
        if x.ndim != 5 or x.size(2) != 3:
            raise ValueError("Input must have shape [B,T,3,H,W]")
        batch, frames, _, height, width = x.shape
        if frames < 1:
            raise ValueError("The frame count T must be at least 1")
        x, frame_order = self._reference_first(x, ref_idx)
        reference_image = x[:, 0]

        flat = x.reshape(batch * frames, 3, height, width)
        pyramid_flat = self.feature(flat)
        pyramids = [
            feature.view(batch, frames, feature.size(1), feature.size(2), feature.size(3))
            for feature in pyramid_flat
        ]
        reference_pyramid = [feature[:, 0] for feature in pyramids]
        flows = self._flows_to_reference(x)

        aligned: List[Tensor] = []
        confidences: List[Tensor] = []
        effective_offsets: List[Tensor] = []
        for index in range(frames):
            if index == 0:
                aligned_feature = reference_pyramid[0]
                confidence = torch.ones(
                    batch, 1, height, width, device=x.device, dtype=x.dtype
                )
                effective_offset = torch.zeros(
                    batch, 2, height, width, device=x.device, dtype=x.dtype
                )
            else:
                aligned_feature, confidence, effective_offset = self.align(
                    [feature[:, index] for feature in pyramids],
                    reference_pyramid,
                    flows[index],
                    x[:, index],
                    reference_image,
                )
            aligned.append(aligned_feature)
            confidences.append(confidence)
            effective_offsets.append(effective_offset)

        aligned_tensor = torch.stack(aligned, dim=1)
        confidence_tensor = torch.stack(confidences, dim=1)
        effective_offset_tensor = torch.stack(effective_offsets, dim=1)
        ref_expand = reference_pyramid[0].unsqueeze(1).expand(-1, frames, -1, -1, -1)
        condition = torch.cat(
            (
                aligned_tensor,
                ref_expand,
                aligned_tensor - ref_expand,
                confidence_tensor,
            ),
            dim=2,
        )
        conditioned = self.condition(
            condition.reshape(batch * frames, -1, height, width)
        ).view(batch, frames, -1, height, width)
        features = aligned_tensor + conditioned

        for block in self.tssrm:
            features = block(features, confidence_tensor, effective_offset_tensor)

        (
            fused,
            fusion_weights,
            phase_latent,
            phase_coverage,
            alignment_reliability,
        ) = self.final_fusion(
            features, confidence_tensor, effective_offset_tensor
        )
        coverage = 1.0 - torch.exp(-phase_coverage.sum(dim=1))
        fused = self.final_spatial(fused, alignment_reliability, coverage)
        reconstruction = self.reconstruction(
            torch.cat((self.local_reference(reference_pyramid[0]), fused), dim=1)
        )
        upsampled = self.upsampler(reconstruction)
        phase_count = self.scale * self.scale
        phase_expanded = self.phase_expand(
            phase_latent.reshape(
                batch * phase_count, self.phase_channels, height, width
            )
        ).view(batch, phase_count, -1, height, width)
        # Reorder [B,P,C,H,W] channel-first so fixed phase slots map directly through PixelShuffle.
        phase_packed = phase_expanded.permute(0, 2, 1, 3, 4).reshape(
            batch, -1, height, width
        )
        phase_hr = self.phase_refine(self.phase_shuffle(phase_packed))
        residual = self.hr(
            upsampled + self.phase_residual_scale * phase_hr
        )
        base = F.interpolate(
            reference_image,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )
        sr = base + residual
        if return_aux:
            return sr, {
                "flows": torch.stack(flows, dim=1),
                "confidence": confidence_tensor,
                "fusion_weights": fusion_weights,
                "aligned": aligned_tensor,
                "effective_offsets": effective_offset_tensor,
                "fractional_phase": torch.remainder(
                    effective_offset_tensor.float(), 1.0
                ).to(x.dtype),
                "phase_coverage": phase_coverage,
                "alignment_reliability": alignment_reliability,
                "frame_order": frame_order,
            }
        return sr
