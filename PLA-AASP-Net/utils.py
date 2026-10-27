"""Checkpoint, image, and device helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def strip_prefix(state: Dict[str, Tensor], prefixes: Sequence[str]) -> Dict[str, Tensor]:
    output: Dict[str, Tensor] = {}
    for key, value in state.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        output[key] = value
    return output

def torch_load_compat(path: str | Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def crop_border(x: Tensor, border: int) -> Tensor:
    if border < 0 or border * 2 >= min(x.shape[-2:]):
        raise ValueError(f"Invalid crop border: {border}")
    return x if border == 0 else x[..., border:-border, border:-border]

def save_image(x: Tensor, path: str | Path) -> None:
    array = x.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB").save(path)

def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch environment")
    return device
