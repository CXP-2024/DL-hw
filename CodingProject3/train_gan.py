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

    from torch.utils.data import DataLoader

    batch_size = 64
    lr_g = 2e-4
    lr_d = 7e-4
    n_epochs = 80
    latent_dim = generator.latent_dim

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=torch.accelerator.is_available(), drop_last=True
    )

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.9))

    # Hinge loss for stable training with spectral norm
    def d_hinge_loss(d_real, d_fake):
        return torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()

    def g_hinge_loss(d_fake):
        return -d_fake.mean()

    for epoch in range(n_epochs):
        total_d_loss = 0.0
        total_g_loss = 0.0
        n_batches = 0

        for images, labels in dataloader:
            bs = images.size(0)
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            real_images = images * 2 - 1

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()

            d_real = discriminator(real_images, labels)

            z = torch.randn(bs, latent_dim, device=DEVICE)
            fake_images = generator(z, labels).detach()
            d_fake = discriminator(fake_images, labels)

            d_loss = d_hinge_loss(d_real, d_fake)
            d_loss.backward()
            optimizer_d.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_g.zero_grad()

            z = torch.randn(bs, latent_dim, device=DEVICE)
            fake_images = generator(z, labels)
            d_fake = discriminator(fake_images, labels)

            g_loss = g_hinge_loss(d_fake)
            g_loss.backward()
            optimizer_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            n_batches += 1

        avg_d = total_d_loss / n_batches
        avg_g = total_g_loss / n_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{n_epochs}, D_loss: {avg_d:.4f}, G_loss: {avg_g:.4f}")

    # YOUR CODE END.


if __name__ == "__main__":
    main()
