import argparse


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    args, remaining = parser.parse_known_args()
    if args.mode == "train":
        from train import main as run
    else:
        from test import main as run
    run(remaining)


if __name__ == "__main__":
    main()
