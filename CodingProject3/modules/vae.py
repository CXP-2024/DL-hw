import torch
from torch import nn


class CustomVAEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # YOUR CODE BEGIN.

        raise NotImplementedError()

        # YOUR CODE END.

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images of shape (batch_size, #channels, height, width).
            labels: Labels of shape (batch_size,).

        Returns:
            - Reconstructed images of shape (batch_size, #channels, height, width).
            - Latent means of shape (batch_size, latent_dim).
            - Latent log variances of shape (batch_size, latent_dim).
        """

        mu, logvar = self._encode(x, labels)
        z = self._reparameterize(mu, logvar)
        x_reconstructed = self._decode(z, labels)

        return x_reconstructed, mu, logvar

    def generate(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: Labels of shape (batch_size,).

        Returns:
            Generated images of shape (batch_size, #channels, height, width)
            with pixel values in [0, 1].
        """

        # YOUR CODE BEGIN.

        raise NotImplementedError()

        # YOUR CODE END.

    def _encode(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # YOUR CODE BEGIN.

        raise NotImplementedError()

        # YOUR CODE END.

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # YOUR CODE BEGIN.

        raise NotImplementedError()

        # YOUR CODE END.

    def _decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # YOUR CODE BEGIN.

        raise NotImplementedError()

        # YOUR CODE END.
