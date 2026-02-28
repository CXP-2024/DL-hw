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
        # YOUR CODE BEGIN. input x, output y, grad dL/dy, return dL/dx
        # if x>0, then dL/dy=dL/dx, else dL/dx=0
        relu_grad = np.where(x > 0, 1.0, 0.0)
        return grad * relu_grad

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
        # dL/dx = dL/dy * dy/dx, where dy/dx = 1 - tanh(x)^2
        tanh_x = np.tanh(x)
        tanh_grad = 1.0 - tanh_x ** 2
        return grad * tanh_grad

        # YOUR CODE END.
