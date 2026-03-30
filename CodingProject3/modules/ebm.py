import torch
from torch import nn


class CustomEBMModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # YOUR CODE BEGIN.

        # Wider MLP with Swish activations
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

        # YOUR CODE END.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (batch_size, #channels, height, width).

        Returns:
            Energy values of shape (batch_size, 1).
        """

        # YOUR CODE BEGIN.

        return self.net(x)

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

        n_steps = 80
        step_size = 0.015

        x = corrupted_images.clone().detach()

        for _ in range(n_steps):
            x.requires_grad_(True)
            energy = self.forward(x).sum()
            energy.backward()
            grad = x.grad.detach()

            with torch.no_grad():
                x = x - step_size * grad
                x = torch.where(mask, corrupted_images, x)
                x = x.clamp(0, 1)
                x = x.detach()

        return x

        # YOUR CODE END.
