import numpy as np
from numpy.typing import NDArray

from modules.module import Module


class MaxPool2d(Module):
    kernel_size: tuple[int, int]
    stride: tuple[int, int]

    def __init__(self, kernel_size: tuple[int, int], stride: tuple[int, int]):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        _, _, h, w = x.shape

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        out_h = (h - kernel_h) // stride_h + 1
        out_w = (w - kernel_w) // stride_w + 1

        idx_h = np.arange(out_h)[:, None] * stride_h + np.arange(kernel_h)
        idx_w = np.arange(out_w)[:, None] * stride_w + np.arange(kernel_w)

        windows = x[:, :, idx_h[:, None, :, None], idx_w[None, :, None, :]]

        out = windows.max(axis=(-2, -1))

        return out

    def backward(
        self, grad: NDArray[np.float32], x: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Backward pass for 2D max pooling.

        Routes the upstream gradient to the positions of the maximum values
        in each pooling window; all other positions receive zero gradient.

        Args:
            grad: Upstream gradient, shape (batch_size, channels, out_h, out_w).
            x: Input from the forward pass, shape (batch_size, channels, h, w).

        Returns:
            Gradient of the loss w.r.t. the input, shape (batch_size, channels, h, w).
        """
        # YOUR CODE BEGIN.
        _, _, h, w = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        out_h = (h - kernel_h) // stride_h + 1
        out_w = (w - kernel_w) // stride_w + 1

        idx_h = np.arange(out_h)[:, None] * stride_h + np.arange(kernel_h)
        idx_w = np.arange(out_w)[:, None] * stride_w + np.arange(kernel_w)

        windows = x[:, :, idx_h[:, None, :, None], idx_w[None, :, None, :]]
        # windows shape: (batch_size, channels, out_h,  out_w, kernel_h, kernel_w)

        # route upstream gradient through the mask
        b, c = x.shape[0], x.shape[1]
        grad_input = np.zeros_like(x)
        for i in range(out_h):
            for j in range(out_w):
                window = windows[:, :, i, j, :, :]
                # flatten spatial dims, find argmax
                flat = window.reshape(b, c, -1)
                max_pos = np.argmax(flat, axis=-1)  # (batch, channels)
                # build mask: 1 at max position, 0 elsewhere
                mask = np.zeros_like(flat)
                mask[np.arange(b)[:, None], np.arange(c)[None, :], max_pos] = 1
                mask = mask.reshape(b, c, kernel_h, kernel_w)
                # route gradient to max positions
                grad_input[:, :, idx_h[i][:, None], idx_w[j][None, :]] += mask * grad[:, :, i:i+1, j:j+1]
        return grad_input
