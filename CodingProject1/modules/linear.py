from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from modules.module import Module


class Linear(Module):
    class Gradients(NamedTuple):
        weight: NDArray[np.float32]
        bias: NDArray[np.float32]

    in_features: int
    out_features: int

    weight: NDArray[np.float32]  # (in_features, out_features)
    bias: NDArray[np.float32]  # (out_features,)

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = np.empty((in_features, out_features), dtype=np.float32)
        self.bias = np.empty(out_features, dtype=np.float32)

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        out = x @ self.weight

        if self.bias is not None:
            out += self.bias

        return out

    def backward(
        self, grad: NDArray[np.float32], x: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], Gradients]:
        """Backward pass for the linear (fully-connected) layer.

        Computes the gradient of the loss w.r.t. the input and the learnable
        parameters (weight and bias).

        Args:
            grad: Upstream gradient, shape (batch_size, out_features).
            x: Input from the forward pass, shape (batch_size, in_features).

        Returns:
            A tuple of:
                - Gradient of the loss w.r.t. the input, shape (batch_size, in_features).
                - Gradients namedtuple containing:
                    - weight: Gradient w.r.t. weight, shape (in_features, out_features).
                    - bias: Gradient w.r.t. bias, shape (out_features,).
        """
        # YOUR CODE BEGIN.
        # First is dL/dx = dL/dy * dy/dx, where dy/dx = weight^T
        grad_input = grad @ self.weight.T

        # Compute gradients w.r.t. weight and bias
        grad_weight = x.T @ grad
        grad_bias = np.sum(grad, axis=0)

        # Return gradient w.r.t. input and gradients namedtuple
        return grad_input, self.Gradients(weight=grad_weight, bias=grad_bias)
