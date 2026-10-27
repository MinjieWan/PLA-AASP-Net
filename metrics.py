from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable if iterable is not None else range(kwargs.get("total", 0))

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs) -> None:
            pass

from torch import Tensor

from dataset import BurstSRDataset
from utils import crop_border, save_image


def rgb_to_y(x: Tensor) -> Tensor:
    """Convert RGB [0,1] to the MATLAB/BT.601 limited-range Y channel in [0,1]."""
    if x.size(-3) != 3:
        raise ValueError("rgb_to_y requires three RGB channels")
    coefficients = x.new_tensor([65.481, 128.553, 24.966]).view(
        *([1] * (x.ndim - 3)), 3, 1, 1
    )
    return (x * coefficients).sum(dim=-3, keepdim=True) / 255.0 + 16.0 / 255.0

def calculate_psnr(
    prediction: Tensor,
    target: Tensor,
    border: int = 0,
    y_only: bool = True,
) -> Tensor:
    prediction = crop_border(prediction.clamp(0, 1), border)
    target = crop_border(target.clamp(0, 1), border)
    if y_only:
        prediction, target = rgb_to_y(prediction), rgb_to_y(target)
    mse = (prediction - target).square().flatten(1).mean(1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))

def calculate_ssim(
    prediction: Tensor,
    target: Tensor,
    border: int = 0,
    y_only: bool = True,
) -> Tensor:
    prediction = crop_border(prediction.clamp(0, 1), border)
    target = crop_border(target.clamp(0, 1), border)
    if y_only:
        prediction, target = rgb_to_y(prediction), rgb_to_y(target)
    channels = prediction.size(1)
    coordinates = torch.arange(11, device=prediction.device, dtype=prediction.dtype) - 5
    kernel_1d = torch.exp(-coordinates.square() / (2 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    window = (kernel_1d[:, None] * kernel_1d[None, :]).expand(channels, 1, 11, 11)
    mean_x = F.conv2d(prediction, window, groups=channels)
    mean_y = F.conv2d(target, window, groups=channels)
    mean_x2, mean_y2, mean_xy = mean_x.square(), mean_y.square(), mean_x * mean_y
    variance_x = F.conv2d(prediction.square(), window, groups=channels) - mean_x2
    variance_y = F.conv2d(target.square(), window, groups=channels) - mean_y2
    covariance = F.conv2d(prediction * target, window, groups=channels) - mean_xy
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_xy + c1) * (2 * covariance + c2)) / (
        (mean_x2 + mean_y2 + c1) * (variance_x + variance_y + c2)
    )
    return score.flatten(1).mean(1)


METRIC_NAMES = (
    "psnr_rgb",
    "psnr_y",
    "ssim_rgb",
    "ssim_y",
    "lpips",
    "bicubic_psnr_rgb",
    "bicubic_psnr_y",
    "bicubic_ssim_rgb",
    "bicubic_ssim_y",
    "bicubic_lpips",
)


def build_lpips(device: torch.device, network: str = "alex") -> torch.nn.Module:
    try:
        import lpips
    except ImportError as error:
        raise ImportError("LPIPS is missing. Install dependencies with: python -m pip install -r requirements.txt") from error
    model = lpips.LPIPS(net=network, version="0.1", verbose=False).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def calculate_lpips(
    prediction: torch.Tensor,
    target: torch.Tensor,
    model: torch.nn.Module,
    border: int,
) -> torch.Tensor:
    prediction = crop_border(prediction.clamp(0, 1), border) * 2.0 - 1.0
    target = crop_border(target.clamp(0, 1), border) * 2.0 - 1.0
    return model(prediction, target).flatten(1).mean(1)


def evaluate_model(
    model: torch.nn.Module,
    dataset: BurstSRDataset,
    device: torch.device,
    amp: bool,
    scale: int,
    ref_index: int,
    metric_border: int,
    lpips_model: torch.nn.Module,
    output: Optional[Path] = None,
    save_images: bool = False,
    save_reference: bool = False,
    write_details: bool = True,
    max_samples: int = 0,
    description: str = "Evaluate",
) -> Tuple[Dict[str, float | int], List[Dict[str, float | str]]]:
    was_training = model.training
    model.eval()
    rows: List[Dict[str, float | str]] = []
    totals = {name: 0.0 for name in METRIC_NAMES}
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        progress = tqdm(range(len(dataset)), desc=description)
        for index in progress:
            sample = dataset[index]
            sample_id = str(sample["id"])
            lr = sample["lr"].unsqueeze(0).to(device)
            hr = sample["hr"].unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=amp):
                sr = model(lr, ref_idx=ref_index)
            sr = sr.float().clamp(0, 1)
            bicubic = F.interpolate(
                lr[:, ref_index],
                scale_factor=scale,
                mode="bicubic",
                align_corners=False,
            ).float().clamp(0, 1)
            values = {
                "psnr_rgb": calculate_psnr(sr, hr, metric_border, False).item(),
                "psnr_y": calculate_psnr(sr, hr, metric_border, True).item(),
                "ssim_rgb": calculate_ssim(sr, hr, metric_border, False).item(),
                "ssim_y": calculate_ssim(sr, hr, metric_border, True).item(),
                "lpips": calculate_lpips(sr, hr, lpips_model, metric_border).item(),
                "bicubic_psnr_rgb": calculate_psnr(
                    bicubic, hr, metric_border, False
                ).item(),
                "bicubic_psnr_y": calculate_psnr(
                    bicubic, hr, metric_border, True
                ).item(),
                "bicubic_ssim_rgb": calculate_ssim(
                    bicubic, hr, metric_border, False
                ).item(),
                "bicubic_ssim_y": calculate_ssim(
                    bicubic, hr, metric_border, True
                ).item(),
                "bicubic_lpips": calculate_lpips(
                    bicubic, hr, lpips_model, metric_border
                ).item(),
            }
            rows.append({"id": sample_id, **values})
            for key, value in values.items():
                totals[key] += value

            if output is not None and save_images:
                save_image(sr[0], output / f"{sample_id}_sr.png")
            if output is not None and save_reference:
                save_image(hr[0], output / f"{sample_id}_hr.png")
                save_image(bicubic[0], output / f"{sample_id}_bicubic.png")

            progress.set_postfix(
                rgb=f"{totals['psnr_rgb'] / len(rows):.2f}",
                y=f"{totals['psnr_y'] / len(rows):.2f}",
                lpips=f"{totals['lpips'] / len(rows):.4f}",
            )
            if max_samples and len(rows) >= max_samples:
                break

    if not rows:
        raise RuntimeError("The evaluation dataset is empty")
    summary: Dict[str, float | int] = {
        key: value / len(rows) for key, value in totals.items()
    }
    summary["samples"] = len(rows)
    summary["psnr_rgb_gain"] = float(summary["psnr_rgb"]) - float(
        summary["bicubic_psnr_rgb"]
    )
    summary["psnr_y_gain"] = float(summary["psnr_y"]) - float(
        summary["bicubic_psnr_y"]
    )
    summary["ssim_rgb_gain"] = float(summary["ssim_rgb"]) - float(
        summary["bicubic_ssim_rgb"]
    )
    summary["ssim_y_gain"] = float(summary["ssim_y"]) - float(
        summary["bicubic_ssim_y"]
    )
    summary["lpips_improvement"] = float(summary["bicubic_lpips"]) - float(
        summary["lpips"]
    )

    if output is not None:
        with (output / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        if write_details:
            with (output / "metrics.csv").open(
                "w", newline="", encoding="utf-8-sig"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    if was_training:
        model.train()
    return summary, rows


def print_summary(summary: Dict[str, float | int]) -> None:
    print(
        f"Model: PSNR-RGB={float(summary['psnr_rgb']):.3f} dB, "
        f"PSNR-Y={float(summary['psnr_y']):.3f} dB, "
        f"SSIM-RGB={float(summary['ssim_rgb']):.5f}, "
        f"SSIM-Y={float(summary['ssim_y']):.5f}, "
        f"LPIPS={float(summary['lpips']):.6f}"
    )
    print(
        f"Bicubic: PSNR-RGB={float(summary['bicubic_psnr_rgb']):.3f} dB, "
        f"PSNR-Y={float(summary['bicubic_psnr_y']):.3f} dB, "
        f"SSIM-RGB={float(summary['bicubic_ssim_rgb']):.5f}, "
        f"SSIM-Y={float(summary['bicubic_ssim_y']):.5f}, "
        f"LPIPS={float(summary['bicubic_lpips']):.6f}"
    )
    print(
        f"Gain over bicubic: RGB-PSNR {float(summary['psnr_rgb_gain']):+.3f} dB, "
        f"Y-PSNR {float(summary['psnr_y_gain']):+.3f} dB, "
        f"LPIPS improvement {float(summary['lpips_improvement']):+.6f}"
    )
