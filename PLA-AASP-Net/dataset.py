from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


SUPPORTED_IMAGE_SUFFIXES = {".png", ".bmp"}


def _natural_key(path: Path) -> List[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _read_rgb(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _collect_lr_frames(burst_dir: Path) -> List[Path]:
    """Collect lr_frame_* images in natural order, accepting PNG/BMP in either case."""
    return sorted(
        (
            path
            for path in burst_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            and path.stem.lower().startswith("lr_frame_")
        ),
        key=_natural_key,
    )


def _index_hr_images(hr_root: Path) -> Dict[str, Path]:
    """Index HR images by stem, preferring PNG over BMP for duplicate names."""
    suffix_priority = {".png": 0, ".bmp": 1}
    indexed: Dict[str, Path] = {}
    for path in hr_root.iterdir():
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        key = path.stem.lower()
        current = indexed.get(key)
        if current is None or suffix_priority[suffix] < suffix_priority[current.suffix.lower()]:
            indexed[key] = path
    return indexed


class BurstSRDataset(Dataset):
    """Pair an LR burst directory with the HR image bearing its sample identifier."""

    def __init__(
        self,
        lr_root: str | Path,
        hr_root: str | Path,
        n_frames: int = 0,
        patch_size: int = 0,
        augment: bool = False,
        sample_ids: Optional[Sequence[str]] = None,
        strict: bool = True,
    ) -> None:
        self.lr_root = Path(lr_root)
        self.hr_root = Path(hr_root)
        self.n_frames = n_frames
        self.patch_size = patch_size
        self.augment = augment
        if not self.lr_root.is_dir() or not self.hr_root.is_dir():
            raise FileNotFoundError(f"Dataset directories do not exist: LR={self.lr_root}, HR={self.hr_root}")

        allowed = set(sample_ids) if sample_ids is not None else None
        hr_images = _index_hr_images(self.hr_root)
        self.samples: List[Tuple[str, List[Path], Path]] = []
        missing: List[str] = []
        for burst_dir in sorted(
            (path for path in self.lr_root.iterdir() if path.is_dir()), key=_natural_key
        ):
            sample_id = burst_dir.name.split("-")[0]
            if allowed is not None and sample_id not in allowed:
                continue
            hr_path = hr_images.get(sample_id.lower())
            frames = _collect_lr_frames(burst_dir)
            if hr_path is None or not frames:
                missing.append(sample_id)
                continue
            if n_frames < 0:
                raise ValueError("n_frames must be nonnegative")
            if n_frames and len(frames) < n_frames:
                missing.append(sample_id)
                continue
            self.samples.append((sample_id, frames, hr_path))

        if strict and missing:
            preview = ", ".join(missing[:8])
            raise RuntimeError(f"{len(missing)} incomplete LR/HR samples, for example: {preview}")
        if not self.samples:
            raise RuntimeError(f"No paired samples found in {self.lr_root} and {self.hr_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _select_frames(self, frames: Sequence[Path]) -> List[Path]:
        if not self.n_frames or self.n_frames == len(frames):
            return list(frames)
        if self.n_frames == 1:
            return [frames[0]]
        # Select evenly spaced frames and retain frame zero, which defines the HR coordinates.
        indices = np.linspace(0, len(frames) - 1, self.n_frames).round().astype(np.int64)
        indices[0] = 0
        return [frames[int(index)] for index in indices]

    def _paired_crop(self, lr: Tensor, hr: Tensor) -> Tuple[Tensor, Tensor]:
        _, _, lr_height, lr_width = lr.shape
        hr_height, hr_width = hr.shape[-2:]
        if (
            hr_height % lr_height
            or hr_width % lr_width
            or hr_height // lr_height != hr_width // lr_width
        ):
            raise ValueError(
                f"Inconsistent LR/HR scale ratios: LR={lr_height}x{lr_width}, HR={hr_height}x{hr_width}"
            )
        scale = hr_height // lr_height
        if not self.patch_size:
            return lr, hr
        if self.patch_size > min(lr_height, lr_width):
            raise ValueError(
                f"patch_size={self.patch_size} exceeds LR dimensions {lr_height}x{lr_width}"
            )
        top = random.randint(0, lr_height - self.patch_size)
        left = random.randint(0, lr_width - self.patch_size)
        lr = lr[:, :, top : top + self.patch_size, left : left + self.patch_size]
        hr = hr[
            :,
            top * scale : (top + self.patch_size) * scale,
            left * scale : (left + self.patch_size) * scale,
        ]
        return lr, hr

    def _paired_augment(self, lr: Tensor, hr: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.augment:
            return lr, hr
        if random.random() < 0.5:
            lr, hr = lr.flip(-1), hr.flip(-1)
        if random.random() < 0.5:
            lr, hr = lr.flip(-2), hr.flip(-2)
        rotations = random.randrange(4)
        if rotations:
            lr = torch.rot90(lr, rotations, (-2, -1))
            hr = torch.rot90(hr, rotations, (-2, -1))
        return lr.contiguous(), hr.contiguous()

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        sample_id, all_frames, hr_path = self.samples[index]
        frame_paths = self._select_frames(all_frames)
        lr = torch.stack([_read_rgb(path) for path in frame_paths])
        hr = _read_rgb(hr_path)
        if any(frame.shape != lr[0].shape for frame in lr):
            raise ValueError(f"Inconsistent LR frame dimensions in sample {sample_id}")
        lr, hr = self._paired_crop(lr, hr)
        lr, hr = self._paired_augment(lr, hr)
        return {"lr": lr, "hr": hr, "id": sample_id}
