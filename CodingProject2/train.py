import logging
from argparse import ArgumentParser
from pathlib import Path

import torch

from datasets import TinyImageNetDataset
from evaluate import DATASET_ROOT, DEVICE, TRANSFORM
from modules import CustomModel


def train(model: CustomModel, dataset: TinyImageNetDataset) -> None:
    # YOUR CODE BEGIN.

    raise NotImplementedError

    # YOUR CODE END.


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    args = parser.parse_args()

    ckpt_path: Path = args.ckpt_path

    dataset = TinyImageNetDataset(DATASET_ROOT, "train", transform=TRANSFORM)

    model = CustomModel().to(DEVICE)

    if sum(p.numel() for p in model.parameters()) > 20_000_000:
        logging.error("Model has more than 20 million parameters")
        return

    train(model, dataset)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    logging.info(f"Model checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
