import torch
from torch import nn


class CustomGANGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # YOUR CODE BEGIN.

        raise NotImplementedError()

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

        raise NotImplementedError()

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

        raise NotImplementedError()

        # YOUR CODE END.


class CustomGANDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # YOUR CODE BEGIN.

        raise NotImplementedError()

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

        raise NotImplementedError()

        # YOUR CODE END.
