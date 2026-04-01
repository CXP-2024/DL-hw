import torch
from torch import nn
import torch.nn.functional as F


class CustomVAEModel(nn.Module):
    """Gaussian Mixture VAE (GM-VAE) with learnable mixture-of-Gaussians prior.

    Latent structure:
        z ~ Categorical(1/K)           discrete cluster assignment
        w ~ N(mu_k, sigma_k^2 I)       continuous latent, conditioned on z=k
        x ~ p_theta(x | w, label)      decoder

    Inference:
        q(w | x, label)      encoder (Gaussian)
        q(z | w, x, label)   cluster inference (categorical)
    """

    def __init__(self) -> None:
        super().__init__()

        self.latent_dim = 64
        self.num_classes = 10
        self.img_dim = 784  # 28*28
        self.K = 20  # number of mixture components

        self.label_emb = nn.Embedding(self.num_classes, 50)

        # --- Encoder q(w | x, label) ---
        self.encoder = nn.Sequential(
            nn.Linear(self.img_dim + 50, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, self.latent_dim)
        self.fc_logvar = nn.Linear(256, self.latent_dim)

        # --- Cluster inference q(z | w, x, label) ---
        self.cluster_net = nn.Sequential(
            nn.Linear(self.latent_dim + self.img_dim + 50, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.K),
        )

        # --- Decoder p(x | w, label) ---
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim + 50, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.img_dim),
            nn.Sigmoid(),
        )

        # --- Learnable prior parameters p(w | z=k) = N(mu_k, sigma_k^2 I) ---
        self.mu_prior = nn.Parameter(torch.randn(self.K, self.latent_dim) * 0.5)
        self.logvar_prior = nn.Parameter(torch.zeros(self.K, self.latent_dim))

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (x_recon, mu, logvar, z_logits)
            z_logits: (batch, K) raw logits for cluster assignment
        """
        mu, logvar = self._encode(x, labels)
        w = self._reparameterize(mu, logvar)
        x_recon = self._decode(w, labels)
        z_logits = self._cluster_infer(w, x, labels)
        return x_recon, mu, logvar, z_logits

    def generate(self, labels: torch.Tensor) -> torch.Tensor:
        batch_size = labels.size(0)
        device = labels.device

        # Sample z ~ Categorical(1/K)
        z_idx = torch.randint(0, self.K, (batch_size,), device=device)

        # Sample w ~ N(mu_prior[z], exp(logvar_prior[z]))
        mu_k = self.mu_prior[z_idx]          # (batch, latent_dim)
        logvar_k = self.logvar_prior[z_idx]   # (batch, latent_dim)
        std_k = torch.exp(0.5 * logvar_k)
        w = mu_k + std_k * torch.randn_like(std_k)

        return self._decode(w, labels)

    def _encode(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_flat = x.view(x.size(0), -1)
        label_emb = self.label_emb(labels)
        h = torch.cat([x_flat, label_emb], dim=1)
        h = self.encoder(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def _decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_emb(labels)
        h = torch.cat([z, label_emb], dim=1)
        x_flat = self.decoder(h)
        return x_flat.view(-1, 1, 28, 28)

    def _cluster_infer(self, w: torch.Tensor, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """q(z | w, x, label) — returns raw logits (batch, K)."""
        x_flat = x.view(x.size(0), -1)
        label_emb = self.label_emb(labels)
        h = torch.cat([w, x_flat, label_emb], dim=1)
        return self.cluster_net(h)
