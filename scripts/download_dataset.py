import argparse
from pathlib import Path
from torchvision.datasets import CIFAR100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", 
        type=Path,
        help="Path to the dataset"
    )
    args = parser.parse_args()

    _ = CIFAR100(
        root=args.data_path,
        download=True
    )


if __name__ == "__main__":
    main()