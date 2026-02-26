import numpy as np
from numpy.typing import NDArray

from modules.module import Module


class Unflatten(Module):
    dim: int
    unflattened_size: tuple[int, ...]

    def __init__(self, dim: int, unflattened_size: tuple[int, ...]):
        super().__init__()

        self.dim = dim
        self.unflattened_size = unflattened_size

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        new_shape = (
            *x.shape[: self.dim],
            *self.unflattened_size,
            *x.shape[self.dim + 1 :],
        )
        return np.reshape(x, new_shape)

    def backward(
        self, grad: NDArray[np.float32], x: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Backward pass for Unflatten.

        Reshapes the upstream gradient back to the shape before unflattening
        (i.e., the shape of the original input to forward).

        Args:
            grad: Upstream gradient, shape matching the output of forward
                  (*x.shape[:dim], *unflattened_size, *x.shape[dim+1:]).
            x: Input from the forward pass, shape (*batch_dims, flattened_dim, *trailing_dims).

        Returns:
            Gradient of the loss w.r.t. the input, same shape as x.
        """
        # YOUR CODE BEGIN.

        raise NotImplementedError

        # YOUR CODE END.
