"""Set dataset paths and training/testing defaults here."""
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPYNET_WEIGHTS = PROJECT_ROOT / "checkpoints" / "spynet_20210409-c6c1bd09.pth"


def parse_train_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PLA-AASP Net and evaluate after every epoch"
    )
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--train-lr", default="folder1")
    parser.add_argument("--train-hr", default="folder3")
    parser.add_argument("--test-lr", default="folder5")
    parser.add_argument("--test-hr", default="folder6")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--test-output", type=Path, default=Path("results"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--spynet-weights",
        type=Path,
        default=DEFAULT_SPYNET_WEIGHTS,
        help="A fresh training run requires compatible pretrained SPyNet weights",
    )

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=0)
    parser.add_argument("--frames", type=int, default=0, help="0 uses all available frames in each sample")
    parser.add_argument("--ref-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--feat-blocks", type=int, default=4)
    parser.add_argument("--psi-blocks", "--tssrm-blocks", dest="tssrm_blocks", type=int, default=1)
    parser.add_argument("--slu-bottleneck", "--tssrm-bottleneck", dest="tssrm_bottleneck", type=int, default=24)
    parser.add_argument("--recon-blocks", type=int, default=4)
    parser.add_argument("--deform-groups", type=int, default=8)
    parser.add_argument("--max-residue", type=float, default=10.0)
    parser.add_argument("--max-flow", type=float, default=32.0)
    parser.add_argument("--phase-channels", type=int, default=8)
    parser.add_argument("--phase-temperature", type=float, default=0.04)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--flow-lr-ratio", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--edge-weight", type=float, default=0.05)
    parser.add_argument("--loss-border", type=int, default=4)
    parser.add_argument("--lr-reduction-factor", type=float, default=0.95)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--freeze-flow-epochs", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-every", type=int, default=10)

    parser.add_argument(
        "--test-every", type=int, default=1, help="Must be 1: evaluate after every epoch"
    )
    parser.add_argument("--metric-border", type=int, default=4)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--test-max-samples", type=int, default=0)
    return parser.parse_args(argv)

def parse_test_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PLA-AASP Net")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best.pth"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--test-lr", default="folder5")
    parser.add_argument("--test-hr", default="folder6")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--ref-index", type=int, default=0)
    parser.add_argument("--metric-border", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--save-reference", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")

    # Used only for a plain state_dict without model_config.
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--feat-blocks", type=int, default=4)
    parser.add_argument("--psi-blocks", "--tssrm-blocks", dest="tssrm_blocks", type=int, default=1)
    parser.add_argument("--slu-bottleneck", "--tssrm-bottleneck", dest="tssrm_bottleneck", type=int, default=24)
    parser.add_argument("--recon-blocks", type=int, default=4)
    parser.add_argument("--deform-groups", type=int, default=8)
    parser.add_argument("--max-residue", type=float, default=10.0)
    parser.add_argument("--max-flow", type=float, default=32.0)
    parser.add_argument("--phase-channels", type=int, default=8)
    parser.add_argument("--phase-temperature", type=float, default=0.04)
    return parser.parse_args(argv)
