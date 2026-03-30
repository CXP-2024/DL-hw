import torch
from torch import nn
from torch.nn.utils import spectral_norm


class CustomGANGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # YOUR CODE BEGIN.

        self.latent_dim = 100
        self.num_classes = 10

        # Label embedding
        self.label_emb = nn.Embedding(self.num_classes, 50)

        # Project (latent + label_emb) to feature map
        self.proj = nn.Linear(self.latent_dim + 50, 256 * 7 * 7)

        self.main = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # (256, 7, 7) -> (128, 14, 14)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (128, 14, 14) -> (1, 28, 28)
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        # YOUR CODE END.

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vectors of shape (batch_size, latent_dim).
            labels: Labels of shape (batch_size,).

        Returns:
            Generated images of shape (batch_size, #channels, height, width).
        """

        # YOUR CODE BEGIN.

        label_input = self.label_emb(labels)
        gen_input = torch.cat([z, label_input], dim=1)
        x = self.proj(gen_input)
        x = x.view(-1, 256, 7, 7)
        x = self.main(x)
        return x

        # YOUR CODE END.

    def generate(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: Labels of shape (batch_size,).

        Returns:
            Generated images of shape (batch_size, #channels, height, width)
            with pixel values in [0, 1].
        """

        # YOUR CODE BEGIN.

        z = torch.randn(labels.size(0), self.latent_dim, device=labels.device)
        images = self.forward(z, labels)
        # Tanh outputs [-1, 1], rescale to [0, 1]
        images = (images + 1) / 2
        return images

        # YOUR CODE END.


class CustomGANDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # YOUR CODE BEGIN.

        self.num_classes = 10

        # Label embedding projected to image size (1, 28, 28)
        self.label_emb = nn.Embedding(self.num_classes, 28 * 28)

        # Use spectral normalization for stability
        self.main = nn.Sequential(
            # (2, 28, 28) -> (64, 14, 14)
            spectral_norm(nn.Conv2d(2, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 14, 14) -> (128, 7, 7)
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 7, 7) -> (1, 1, 1)
            spectral_norm(nn.Conv2d(128, 1, 7, 1, 0)),
        )

        # YOUR CODE END.

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (batch_size, #channels, height, width).
            labels: Labels of shape (batch_size,).

        Returns:
            Discrimination scores of shape (batch_size, 1).
        """

        # YOUR CODE BEGIN.

        label_map = self.label_emb(labels).view(-1, 1, 28, 28)
        x = torch.cat([x, label_map], dim=1)
        output = self.main(x)
        return output.view(-1, 1)

        # YOUR CODE END.
