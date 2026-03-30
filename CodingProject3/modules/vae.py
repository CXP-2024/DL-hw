import torch
from torch import nn


class CustomVAEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # YOUR CODE BEGIN.

        self.latent_dim = 64
        self.num_classes = 10
        self.img_dim = 784  # 28*28

        self.label_emb = nn.Embedding(self.num_classes, 50)

        # Encoder
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

        # Decoder
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

        # YOUR CODE END.

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self._encode(x, labels)
        z = self._reparameterize(mu, logvar)
        x_reconstructed = self._decode(z, labels)
        return x_reconstructed, mu, logvar

    def generate(self, labels: torch.Tensor) -> torch.Tensor:
        # YOUR CODE BEGIN.
        batch_size = labels.size(0)
        z = torch.randn(batch_size, self.latent_dim, device=labels.device)
        images = self._decode(z, labels)
        return images
        # YOUR CODE END.

    def _encode(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # YOUR CODE BEGIN.
        x_flat = x.view(x.size(0), -1)
        label_emb = self.label_emb(labels)
        h = torch.cat([x_flat, label_emb], dim=1)
        h = self.encoder(h)
        return self.fc_mu(h), self.fc_logvar(h)
        # YOUR CODE END.

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # YOUR CODE BEGIN.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
        # YOUR CODE END.

    def _decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # YOUR CODE BEGIN.
        label_emb = self.label_emb(labels)
        h = torch.cat([z, label_emb], dim=1)
        x_flat = self.decoder(h)
        return x_flat.view(-1, 1, 28, 28)
        # YOUR CODE END.
