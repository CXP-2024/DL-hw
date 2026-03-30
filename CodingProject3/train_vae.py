import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Final

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from modules.vae import CustomVAEModel
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

    model = CustomVAEModel().to(DEVICE)

    dataset = MNIST(root="data", train=True, transform=TRANSFORM, download=False)

    train(model, dataset)

    model.cpu()

    ckpt_path: Path = args.ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    logger.info(f"Model checkpoint saved to {ckpt_path}")


def train(model: CustomVAEModel, dataset: MNIST) -> None:
    model.train()

    # YOUR CODE BEGIN.

    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    batch_size = 128
    lr = 1e-3
    n_epochs = 150

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=torch.accelerator.is_available(), drop_last=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    for epoch in range(n_epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        # KL annealing from 0 to 1 over first 30 epochs
        kl_weight = min(1.0, epoch / 30.0)

        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            x_recon, mu, logvar = model(images, labels)

            # BCE reconstruction loss
            recon_loss = F.binary_cross_entropy(x_recon, images, reduction='sum') / images.size(0)

            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)

            loss = recon_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches

        if (epoch + 1) % 15 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")

    # YOUR CODE END.


if __name__ == "__main__":
    main()
