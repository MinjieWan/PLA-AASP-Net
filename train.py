from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

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

from config import parse_train_args as parse_args
from dataset import BurstSRDataset
from loss import CharbonnierEdgeLoss
from utils import choose_device, crop_border, torch_load_compat
from metrics import build_lpips, evaluate_model, print_summary
from models import PLAAASPNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class OutputLock:
    """Prevent concurrent training processes from overwriting checkpoints and logs."""

    def __init__(self, output: Path):
        self.path = output / ".training.lock"
        self.file_descriptor: Optional[int] = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.file_descriptor = os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError as error:
            raise RuntimeError(
                f"Training lock found at {self.path}. Check for other runs using this output directory. "
                "If a previous run crashed and has exited, remove the stale lock manually."
            ) from error
        os.write(self.file_descriptor, str(os.getpid()).encode("ascii"))
        atexit.register(self.release)

    def release(self) -> None:
        if self.file_descriptor is not None:
            os.close(self.file_descriptor)
            self.file_descriptor = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def model_config_from_args(args: argparse.Namespace) -> Dict[str, object]:
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
        "spynet_pretrained": str(args.spynet_weights) if args.spynet_weights else None,
    }


def make_train_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    dataset = BurstSRDataset(
        args.data_root / args.train_lr,
        args.data_root / args.train_hr,
        n_frames=args.frames,
        patch_size=args.patch_size,
        augment=True,
    )
    common = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    print(
        f"Training samples: {len(dataset)}, frames per sample: "
        f"{args.frames if args.frames else 'all'}"
    )
    return loader


def make_test_dataset(args: argparse.Namespace) -> BurstSRDataset:
    dataset = BurstSRDataset(
        args.data_root / args.test_lr,
        args.data_root / args.test_hr,
        n_frames=args.frames,
        patch_size=0,
        augment=False,
    )
    print(f"Evaluation samples: {len(dataset)}; evaluation follows every training epoch")
    return dataset


def make_optimizer(model: PLAAASPNet, args: argparse.Namespace) -> AdamW:
    groups = model.parameter_groups(args.lr, args.flow_lr_ratio)
    return AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))


def save_checkpoint(
    path: Path,
    model: PLAAASPNet,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_train_loss: float,
    no_improvement_count: int,
    model_config: Dict[str, object],
    args: argparse.Namespace,
    test_summary: Optional[Dict[str, float | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_config = dict(model_config)
    clean_config["spynet_pretrained"] = None
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_train_loss": best_train_loss,
            "no_improvement_count": no_improvement_count,
            "model_config": clean_config,
            "test_summary": test_summary,
            "train_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_args(args: argparse.Namespace) -> None:
    if args.accum_steps < 1:
        raise ValueError("accum_steps must be at least 1")
    if args.frames < 0:
        raise ValueError("frames must be nonnegative")
    if args.ref_index != 0:
        raise ValueError("HR targets correspond to frame zero; ref-index must be 0")
    if args.patience < 1:
        raise ValueError("patience must be at least 1")
    if args.test_every != 1:
        raise ValueError("Evaluation runs after every epoch; test-every must be 1")
    if not 0.0 < args.lr_reduction_factor < 1.0:
        raise ValueError("lr-reduction-factor must be in (0,1)")
    if args.phase_channels < 4:
        raise ValueError("phase-channels must be at least 4")
    if args.phase_temperature <= 0:
        raise ValueError("phase-temperature must be positive")
    if args.resume is None and (
        args.spynet_weights is None or not args.spynet_weights.is_file()
    ):
        raise FileNotFoundError(
            f"Fresh training requires a valid pretrained SPyNet checkpoint: {args.spynet_weights}"
        )


def main(argv=None) -> None:
    args = parse_args(argv)
    validate_args(args)
    seed_everything(args.seed)
    device = choose_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    lock = OutputLock(args.output)
    lock.acquire()
    try:
        resume_obj = torch_load_compat(args.resume, device) if args.resume else None
        config = model_config_from_args(args)
        if isinstance(resume_obj, dict) and "model_config" in resume_obj:
            config = dict(resume_obj["model_config"])
        model = PLAAASPNet(**config).to(device)
        optimizer = make_optimizer(model, args)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        criterion = CharbonnierEdgeLoss(args.edge_weight).to(device)
        start_epoch, best_train_loss, no_improvement_count = 1, math.inf, 0

        if resume_obj is not None:
            state = (
                resume_obj.get("model", resume_obj)
                if isinstance(resume_obj, dict)
                else resume_obj
            )
            model.load_state_dict(state, strict=True)
            if isinstance(resume_obj, dict) and "optimizer" in resume_obj:
                optimizer.load_state_dict(resume_obj["optimizer"])
                if resume_obj.get("scaler"):
                    scaler.load_state_dict(resume_obj["scaler"])
                start_epoch = int(resume_obj.get("epoch", 0)) + 1
                best_train_loss = float(resume_obj.get("best_train_loss", math.inf))
                no_improvement_count = int(
                    resume_obj.get("no_improvement_count", 0)
                )
            print(f"Resuming {args.resume} at epoch {start_epoch}")

        train_loader = make_train_loader(args, device)
        test_dataset = make_test_dataset(args)
        lpips_model = build_lpips(device, args.lpips_net)
        args.output.mkdir(parents=True, exist_ok=True)
        args.test_output.mkdir(parents=True, exist_ok=True)
        with (args.output / "config.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"Device: {device}, AMP: {amp}, parameters: {parameters / 1e6:.2f}M, "
            "Schedule: complete evaluation after every training epoch"
        )

        flow_can_freeze = (
            args.spynet_weights is not None
            and args.freeze_flow_epochs > 0
            and resume_obj is None
        )
        history_path = args.output / "metrics.jsonl"
        test_history_path = args.output / "test_history.jsonl"

        for epoch in range(start_epoch, args.epochs + 1):
            flow_frozen = flow_can_freeze and epoch <= args.freeze_flow_epochs
            if flow_frozen:
                model.freeze_spynet()
            else:
                model.unfreeze_spynet()
            model.train()
            if flow_frozen:
                model.spynet.eval()

            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            seen = 0
            train_started = time.time()
            progress = tqdm(
                enumerate(train_loader, 1),
                total=len(train_loader),
                desc=f"Train {epoch}/{args.epochs}",
            )
            for step, batch in progress:
                lr = batch["lr"].to(device, non_blocking=True)
                hr = batch["hr"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    sr = model(lr, ref_idx=args.ref_index)
                    loss = criterion(
                        crop_border(sr, args.loss_border),
                        crop_border(hr, args.loss_border),
                    )
                    scaled_loss = loss / args.accum_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Nonfinite loss for samples={batch['id']}; stopping training"
                    )
                scaler.scale(scaled_loss).backward()
                if step % args.accum_steps == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                batch_size = lr.size(0)
                epoch_loss += loss.item() * batch_size
                seen += batch_size
                progress.set_postfix(
                    loss=f"{epoch_loss / seen:.5f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

            train_seconds = time.time() - train_started
            average_train_loss = epoch_loss / seen
            improved = average_train_loss < best_train_loss
            lr_reduced = False
            if improved:
                best_train_loss = average_train_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= args.patience:
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] *= args.lr_reduction_factor
                    no_improvement_count = 0
                    lr_reduced = True

            test_summary: Optional[Dict[str, float | int]] = None
            test_seconds = 0.0
            should_test = True
            if should_test:
                test_started = time.time()
                is_final_epoch = epoch == args.epochs
                epoch_output = (
                    args.test_output
                    if is_final_epoch
                    else args.output / "epoch_tests" / f"epoch_{epoch:04d}"
                )
                test_summary, _ = evaluate_model(
                    model=model,
                    dataset=test_dataset,
                    device=device,
                    amp=amp,
                    scale=int(config["scale"]),
                    ref_index=args.ref_index,
                    metric_border=args.metric_border,
                    lpips_model=lpips_model,
                    output=epoch_output,
                    save_images=False,
                    save_reference=False,
                    write_details=is_final_epoch,
                    max_samples=args.test_max_samples,
                    description=f"Evaluate {epoch}/{args.epochs}",
                )
                test_seconds = time.time() - test_started
                test_record = {"epoch": epoch, "seconds": test_seconds, **test_summary}
                append_jsonl(test_history_path, test_record)
                print_summary(test_summary)

            record = {
                "epoch": epoch,
                "train_seconds": train_seconds,
                "test_seconds": test_seconds,
                "train_loss": average_train_loss,
                "best_train_loss": best_train_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_reduced": lr_reduced,
                "no_improvement_count": no_improvement_count,
                "test": test_summary,
            }
            append_jsonl(history_path, record)
            print(
                f"epoch={epoch} | train_loss={average_train_loss:.6f} | "
                f"best={best_train_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.8f} | "
                f"train={train_seconds:.1f}s | test={test_seconds:.1f}s"
            )
            if lr_reduced:
                print(
                    f"Training loss did not improve for {args.patience} epoch(s); "
                    f"multiplying learning rates by {args.lr_reduction_factor}"
                )

            if improved:
                save_checkpoint(
                    args.output / "best.pth",
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_train_loss,
                    no_improvement_count,
                    config,
                    args,
                    test_summary,
                )
                print(f"Saved the best training-loss checkpoint: {best_train_loss:.6f}")
            save_checkpoint(
                args.output / "latest.pth",
                model,
                optimizer,
                scaler,
                epoch,
                best_train_loss,
                no_improvement_count,
                config,
                args,
                test_summary,
            )
            if epoch % args.save_every == 0:
                save_checkpoint(
                    args.output / f"epoch_{epoch:04d}.pth",
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_train_loss,
                    no_improvement_count,
                    config,
                    args,
                    test_summary,
                )
    finally:
        lock.release()


if __name__ == "__main__":
    main()
