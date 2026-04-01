import logging
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Final

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from tqdm import tqdm

from CodingProject3.gmvae import CustomVAEModel
from utils import DEVICE

# Run : python train_gmvae.py checkpoints/gmvae.pth

logger = logging.getLogger(__name__)

TRANSFORM: Final = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    """KL(q || p) for diagonal Gaussians, per-sample sum over latent dims.

    All inputs: (batch, latent_dim)
    Returns: (batch,)
    """
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    return 0.5 * (
        (logvar_p - logvar_q)  # log(sigma_p / sigma_q)
        + var_q / var_p
        + (mu_q - mu_p).pow(2) / var_p
        - 1.0
    ).sum(dim=1)


def gmvae_loss(x, x_recon, mu, logvar, z_logits, mu_prior, logvar_prior):
    """Compute negative ELBO for GM-VAE.

    Three terms:
      1) Reconstruction: BCE(x_recon, x)
      2) w-regularization: E_q(z|w,x)[ -KL(q(w|x) || p(w|z)) ]
      3) z-regularization: -KL(q(z|w,x) || Uniform(1/K))

    Returns (loss, recon, kl_w, kl_z) — all per-sample averaged.
    """
    batch_size = x.size(0)
    K = z_logits.size(1)

    # --- Term 1: Reconstruction ---
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum') / batch_size

    # --- Term 2: w-regularization ---
    # q(z|w,x) as probabilities
    q_z = F.softmax(z_logits, dim=1)  # (batch, K)

    # Compute KL(q(w|x) || p(w|z=k)) for each k
    # mu, logvar: (batch, latent_dim)
    # mu_prior, logvar_prior: (K, latent_dim)
    kl_per_k = []
    for k in range(K):
        mu_p_k = mu_prior[k].unsqueeze(0)         # (1, latent_dim)
        logvar_p_k = logvar_prior[k].unsqueeze(0)  # (1, latent_dim)
        kl_k = gaussian_kl(mu, logvar, mu_p_k, logvar_p_k)  # (batch,)
        kl_per_k.append(kl_k)

    kl_per_k = torch.stack(kl_per_k, dim=1)  # (batch, K)
    # Weighted sum: E_q(z)[KL] = sum_k q(z=k) * KL_k
    kl_w = (q_z * kl_per_k).sum(dim=1).mean()

    # --- Term 3: z-regularization ---
    # KL(q(z|w,x) || Uniform(1/K)) = sum_k q_k * log(q_k * K)
    #                                = sum_k q_k * log(q_k) + log(K)
    log_q_z = F.log_softmax(z_logits, dim=1)  # (batch, K)
    kl_z = (q_z * log_q_z).sum(dim=1).mean() + math.log(K)

    loss = recon_loss + kl_w + kl_z
    return loss, recon_loss, kl_w, kl_z


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    args = parser.parse_args()

    model = CustomVAEModel().to(DEVICE)
    logger.info(f"GM-VAE: K={model.K}, latent_dim={model.latent_dim}")
    logger.info(f"Device: {DEVICE}")

    dataset = MNIST(root="data", train=True, transform=TRANSFORM, download=True)

    train(model, dataset)

    model.cpu()
    ckpt_path: Path = args.ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Model checkpoint saved to {ckpt_path}")


def train(model: CustomVAEModel, dataset: MNIST) -> None:
    model.train()

    batch_size = 128
    lr = 1e-3
    n_epochs = 150

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5
    )

    epoch_pbar = tqdm(range(n_epochs), desc="Total", unit="epoch")
    for epoch in epoch_pbar:
        total_loss = 0.0
        total_recon = 0.0
        total_kl_w = 0.0
        total_kl_z = 0.0
        n_batches = 0

        # KL annealing: linear ramp 0 → 1 over first 30 epochs
        kl_weight = min(1.0, epoch / 30.0)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            x_recon, mu, logvar, z_logits = model(images, labels)

            loss_full, recon, kl_w, kl_z = gmvae_loss(
                images, x_recon, mu, logvar, z_logits,
                model.mu_prior, model.logvar_prior,
            )

            # Apply KL annealing to both KL terms
            loss = recon + kl_weight * (kl_w + kl_z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl_w += kl_w.item()
            total_kl_z += kl_z.item()
            n_batches += 1
            pbar.set_postfix(
                L=f"{loss.item():.1f}",
                R=f"{recon.item():.1f}",
                Kw=f"{kl_w.item():.1f}",
                Kz=f"{kl_z.item():.2f}",
            )

        scheduler.step()

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kl_w = total_kl_w / n_batches
        avg_kl_z = total_kl_z / n_batches
        epoch_pbar.set_postfix(
            L=f"{avg_loss:.1f}", R=f"{avg_recon:.1f}",
            Kw=f"{avg_kl_w:.1f}", Kz=f"{avg_kl_z:.2f}",
        )


if __name__ == "__main__":
    main()
