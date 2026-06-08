import argparse
from pathlib import Path
from torchvision.datasets import CIFAR100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", 
        type=Path,
        help="Path to the dataset"
    )
    args = parser.parse_args()

    dataset = CIFAR100(
        root=args.data_path,
        download=True
    )