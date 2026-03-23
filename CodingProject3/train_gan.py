import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Final

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from modules.gan import CustomGANDiscriminator, CustomGANGenerator
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

    generator = CustomGANGenerator().to(DEVICE)
    discriminator = CustomGANDiscriminator().to(DEVICE)

    dataset = MNIST(root="data", download=True, train=True, transform=TRANSFORM)

    train(generator, discriminator, dataset)

    generator.cpu()
    discriminator.cpu()

    ckpt_path: Path = args.ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), ckpt_path)

    logger.info(f"Model checkpoint saved to {ckpt_path}")


def train(
    generator: CustomGANGenerator, discriminator: CustomGANDiscriminator, dataset: MNIST
) -> None:
    generator.train()
    discriminator.train()

    # YOUR CODE BEGIN.

    raise NotImplementedError()

    # YOUR CODE END.


if __name__ == "__main__":
    main()
