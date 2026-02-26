import numpy as np
from numpy.typing import NDArray

from modules.module import Module


class ReLU(Module):
    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.maximum(0.0, x)

    def backward(
        self, grad: NDArray[np.float32], x: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Backward pass for ReLU activation.

        Computes the gradient of the loss w.r.t. the input by masking out
        gradients where the input was non-positive.

        Args:
            grad: Upstream gradient, shape (batch_size, *features).
            x: Input from the forward pass, shape (batch_size, *features).

        Returns:
            Gradient of the loss w.r.t. the input, shape (batch_size, *features).
        """
        # YOUR CODE BEGIN.

        raise NotImplementedError

        # YOUR CODE END.


class Tanh(Module):
    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.tanh(x)

    def backward(
        self, grad: NDArray[np.float32], x: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Backward pass for Tanh activation.

        Computes the gradient of the loss w.r.t. the input using the
        derivative of tanh: 1 - tanh(x)^2.

        Args:
            grad: Upstream gradient, shape (batch_size, *features).
            x: Input from the forward pass, shape (batch_size, *features).

        Returns:
            Gradient of the loss w.r.t. the input, shape (batch_size, *features).
        """
        # YOUR CODE BEGIN.

        raise NotImplementedError

        # YOUR CODE END.
