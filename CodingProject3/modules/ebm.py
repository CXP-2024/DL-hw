import torch
from torch import nn


class CustomEBMModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # YOUR CODE BEGIN.

        raise NotImplementedError()

        # YOUR CODE END.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (batch_size, #channels, height, width).

        Returns:
            Energy values of shape (batch_size, 1).
        """

        # YOUR CODE BEGIN.

        raise NotImplementedError()

        # YOUR CODE END.

    def inpaint(
        self, corrupted_images: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Inpaints corrupted images.

        Args:
            corrupted_images: Corrupted images of shape
                (batch_size, #channels, height, width) with pixel values in
                [0, 1].
            mask: Binary masks of shape (batch_size, #channels, height, width)
                where False indicates corrupted pixels.

        Returns:
            Recovered images of shape (batch_size, #channels, height, width)
            with pixel values in [0, 1].
        """

        # YOUR CODE BEGIN.

        raise NotImplementedError()

        # YOUR CODE END.
