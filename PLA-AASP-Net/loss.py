"""Training loss for PLA-AASP Net."""
from typing import Tuple

import torch
from torch import Tensor, nn

class CharbonnierEdgeLoss(nn.Module):
    def __init__(self, edge_weight: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.edge_weight = edge_weight
        self.eps = eps

    @staticmethod
    def gradient(x: Tensor) -> Tuple[Tensor, Tensor]:
        return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        pixel = torch.sqrt((prediction - target).square() + self.eps).mean()
        pred_y, pred_x = self.gradient(prediction)
        target_y, target_x = self.gradient(target)
        edge = (pred_y - target_y).abs().mean() + (pred_x - target_x).abs().mean()
        return pixel + self.edge_weight * edge
