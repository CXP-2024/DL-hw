import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Final

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from modules.ebm import CustomEBMModel
from utils import DEVICE, ensure_reproducibility

logger = logging.getLogger(__name__)

TRANSFORM: Final = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    args = parser.parse_args()

    ensure_reproducibility()

    model = CustomEBMModel().to(DEVICE)
    model.load_state_dict(
        torch.load(args.ckpt_path, map_location=DEVICE, weights_only=True)
    )

    dataset = MNIST(root="data", train=False, transform=TRANSFORM, download=True)

    mse = evaluate(model, dataset)

    logger.info(f"MSE: {mse}")


def evaluate(model: CustomEBMModel, dataset: MNIST) -> float:
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=512,
        num_workers=4,
        pin_memory=torch.accelerator.is_available(),
    )

    squared_error_sum = 0.0
    num_corrupted_pixels = 0

    for images, _ in dataloader:
        images: torch.Tensor

        images = images.to(DEVICE)

        corrupted_images, mask = _corrupt_images(images)

        recovered_images = model.inpaint(corrupted_images, mask)

        corrupted_region = ~mask
        squared_error = (recovered_images - images).square()

        squared_error_sum += squared_error[corrupted_region].sum().item()
        num_corrupted_pixels += corrupted_region.sum().item()

    return squared_error_sum / num_corrupted_pixels


def _corrupt_images(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[..., ::2, :] = True

    noise = 0.3 * torch.randn_like(x)

    corrupted_images = torch.where(mask, x, noise).clamp(0, 1)

    return corrupted_images, mask


if __name__ == "__main__":
    main()
