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

    dataset = MNIST(root="data", train=True, transform=TRANSFORM, download=False)

    train(model, dataset)

    model.cpu()

    ckpt_path: Path = args.ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    logger.info(f"Model checkpoint saved to {ckpt_path}")


def corrupt_images(x):
    """Same corruption as evaluate_ebm.py"""
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[..., ::2, :] = True
    noise = 0.3 * torch.randn_like(x)
    corrupted = torch.where(mask, x, noise).clamp(0, 1)
    return corrupted, mask


def train(model: CustomEBMModel, dataset: MNIST) -> None:
    model.train()

    # YOUR CODE BEGIN.

    from torch.utils.data import DataLoader

    batch_size = 128
    lr = 1e-3
    n_epochs = 100

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=torch.accelerator.is_available(), drop_last=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0

        for images, _ in dataloader:
            images = images.to(DEVICE)

            # Corrupt images exactly as evaluation does
            corrupted, mask = corrupt_images(images)
            corrupted_region = (~mask).float()

            # Compute gradient of energy at corrupted image
            corrupted.requires_grad_(True)
            energy = model(corrupted).sum()
            grad_energy = torch.autograd.grad(energy, corrupted, create_graph=True)[0]

            # Target: gradient should point from corrupted toward clean
            # ∇E(x_corrupted) = (x_corrupted - x_clean)
            target = corrupted.detach() - images

            # Only compute loss on corrupted pixels
            dsm_loss = ((grad_energy - target) * corrupted_region).square().sum() / corrupted_region.sum()

            # Also add uniform noise DSM for generalization
            sigma = 0.2 + 0.3 * torch.rand(1, device=DEVICE)
            noise = sigma * torch.randn_like(images)
            x_noisy = (images + noise).clamp(0, 1)
            x_noisy.requires_grad_(True)
            energy2 = model(x_noisy).sum()
            grad2 = torch.autograd.grad(energy2, x_noisy, create_graph=True)[0]
            target2 = (x_noisy.detach() - images) / (sigma ** 2)
            dsm_loss2 = ((grad2 - target2) ** 2).mean()

            loss = dsm_loss + 0.1 * dsm_loss2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    # YOUR CODE END.


if __name__ == "__main__":
    main()
