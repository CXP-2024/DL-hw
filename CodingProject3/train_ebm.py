import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Final

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from modules.ebm import CustomEBMModel
from utils import DEVICE

logger = logging.getLogger(__name__)

TRANSFORM: Final = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # Scale to [0, 1].
    ]
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    args = parser.parse_args()

    model = CustomEBMModel().to(DEVICE)

    dataset = MNIST(root="data", train=True, transform=TRANSFORM, download=True)

    train(model, dataset)

    model.cpu()

    ckpt_path: Path = args.ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    logger.info(f"Model checkpoint saved to {ckpt_path}")


def train(model: CustomEBMModel, dataset: MNIST) -> None:
    model.train()

    # YOUR CODE BEGIN.

    raise NotImplementedError()

    # YOUR CODE END.


if __name__ == "__main__":
    main()
