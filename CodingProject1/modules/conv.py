from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from modules.module import Module


class Conv2d(Module):
    class Gradients(NamedTuple):
        weight: NDArray[np.float32]
        bias: NDArray[np.float32]

    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]

    weight: NDArray[
        np.float32
    ]  # (out_channels, in_channels, kernel_size[0], kernel_size[1])
    bias: NDArray[np.float32]  # (out_channels,)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weight = np.empty(
            (out_channels, in_channels, kernel_size[0], kernel_size[1]),
            dtype=np.float32,
        )
        self.bias = np.empty(out_channels, dtype=np.float32)

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        _, _, h, w = x.shape

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        out_h = (h - kernel_h) // stride_h + 1
        out_w = (w - kernel_w) // stride_w + 1

        idx_h = np.arange(kernel_h)[:, None] + np.arange(out_h)[None, :] * stride_h
        idx_w = np.arange(kernel_w)[:, None] + np.arange(out_w)[None, :] * stride_w

        windows = x[:, :, idx_h[:, None, :, None], idx_w[None, :, None, :]]

        out = np.einsum("bc yx hw, ocyx -> bohw", windows, self.weight)

        out = out + self.bias[None, :, None, None]

        return out

    def backward(
        self, grad: NDArray[np.float32], x: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], Gradients]:
        """Backward pass for 2D convolution.

        Computes the gradient of the loss w.r.t. the input and the learnable
        parameters (weight and bias).

        Args:
            grad: Upstream gradient, shape (batch_size, out_channels, out_h, out_w).
            x: Input from the forward pass, shape (batch_size, in_channels, h, w).

        Returns:
            A tuple of:
                - Gradient of the loss w.r.t. the input, shape (batch_size, in_channels, h, w).
                - Gradients namedtuple containing:
                    - weight: Gradient w.r.t. weight, shape (out_channels, in_channels, kernel_size[0], kernel_size[1]).
                    - bias: Gradient w.r.t. bias, shape (out_channels,).
        """
        # YOUR CODE BEGIN.
        # Compute gradients w.r.t. weight and bias
        _, _, h, w = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        out_h = (h - kernel_h) // stride_h + 1
        out_w = (w - kernel_w) // stride_w + 1
        idx_h = np.arange(kernel_h)[:, None] + np.arange(out_h)[None, :] * stride_h
        idx_w = np.arange(kernel_w)[:, None] + np.arange(out_w)[None, :] * stride_w
        windows = x[:, :, idx_h[:, None, :, None], idx_w[None, :, None, :]]
        # windows shape: (batch_size, in_channels, kernel_h, kernel_w, out_h, out_w)

        # For grad_weight, we want to sum over the batch dimension and the output spatial dimensions, grad_weight(out_channels, in_channels, kernel_h, kernel_w) = sum_{b, out_h, out_w} grad(b, out_channels, out_h, out_w) * windows(b, in_channels, kernel_h, kernel_w, out_h, out_w)
        # We can use einsum to express this:
        grad_weight = np.einsum("bohw, bc yx hw -> ocyx", grad, windows)

        # Compute gradient w.r.t. bias
        grad_bias = np.sum(grad, axis=(0, 2, 3))  # sum over batch, out_h, out_w dimensions

        # Compute gradient w.r.t. input, add each input window's contribution to the appropriate location in the input gradient
        grad_input = np.zeros_like(x)
        for i in range(out_h):
            for j in range(out_w):
                hs, ws = i * stride_h, j * stride_w
                g = grad[:, :, i:i+1, j:j+1]
                # g shape: (batch_size, out_channels, 1, 1)
                # compute gradient w.r.t. input for this window
                grad_input[:, :, hs:hs+kernel_h, ws:ws+kernel_w] += np.einsum("bohw, ocyx -> bc yx", g, self.weight)
                # weight shape: (out_channels, in_channels, kernel_h, kernel_w)
                # grad_input shape: (batch_size, in_channels, kernel_h, kernel_w)

        return grad_input, self.Gradients(weight=grad_weight, bias=grad_bias)
