from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from config import parse_test_args as parse_args
from dataset import BurstSRDataset
from metrics import build_lpips, evaluate_model, print_summary
from models import PLAAASPNet
from utils import choose_device, torch_load_compat


def fallback_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "c": args.channels,
        "scale": args.scale,
        "feat_blocks": args.feat_blocks,
        "tssrm_blocks": args.tssrm_blocks,
        "tssrm_bottleneck": args.tssrm_bottleneck,
        "recon_blocks": args.recon_blocks,
        "deform_groups": args.deform_groups,
        "max_residue": args.max_residue,
        "max_flow": args.max_flow,
        "phase_channels": args.phase_channels,
        "phase_temperature": args.phase_temperature,
        "spynet_pretrained": None,
    }


def extract_state(obj) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError("No state_dict found in the checkpoint")
    for key in ("model", "state_dict", "params", "generator"):
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    state: Dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        if not isinstance(value, torch.Tensor):
            continue
        while key.startswith("module."):
            key = key[len("module."):]
        state[key] = value
    if not state:
        raise TypeError("No model tensors found in the checkpoint")
    return state


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.frames < 0:
        raise ValueError("frames must be nonnegative")
    if args.ref_index != 0:
        raise ValueError("HR targets correspond to frame zero; ref-index must be 0")
    device = choose_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    checkpoint = torch_load_compat(args.checkpoint, device)
    config = (
        dict(checkpoint.get("model_config", fallback_config(args)))
        if isinstance(checkpoint, dict)
        else fallback_config(args)
    )
    config["spynet_pretrained"] = None
    model = PLAAASPNet(**config).to(device)
    model.load_state_dict(extract_state(checkpoint), strict=True)
    model.eval()
    lpips_model = build_lpips(device, args.lpips_net)

    dataset = BurstSRDataset(
        args.data_root / args.test_lr,
        args.data_root / args.test_hr,
        n_frames=args.frames,
        patch_size=0,
        augment=False,
    )
    print(f"Device: {device}, Evaluation samples: {len(dataset)}, checkpoint: {args.checkpoint}")
    summary, _ = evaluate_model(
        model=model,
        dataset=dataset,
        device=device,
        amp=amp,
        scale=int(config["scale"]),
        ref_index=args.ref_index,
        metric_border=args.metric_border,
        lpips_model=lpips_model,
        output=args.output,
        save_images=not args.no_save,
        save_reference=args.save_reference,
        write_details=True,
        max_samples=args.max_samples,
        description="Evaluation",
    )
    print_summary(summary)
    print(f"Results directory: {args.output.resolve()}")


if __name__ == "__main__":
    main()
