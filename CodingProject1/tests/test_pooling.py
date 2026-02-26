import unittest

import numpy as np
import torch

from modules.pooling import MaxPool2d


class TestMaxPool2d(unittest.TestCase):
    def test_forward(self) -> None:
        batch_size = 2
        in_channels = 3
        height = 10
        width = 10
        kernel_size = (2, 2)
        stride = (2, 2)

        x_np = np.random.randn(batch_size, in_channels, height, width).astype(
            np.float32
        )

        # Custom implementation
        maxpool = MaxPool2d(kernel_size, stride)
        out_custom = maxpool.forward(x_np)

        # PyTorch implementation
        maxpool_torch = torch.nn.MaxPool2d(kernel_size, stride=stride)
        x_torch = torch.from_numpy(x_np)
        out_torch = maxpool_torch(x_torch)

        np.testing.assert_allclose(out_custom, out_torch.numpy(), rtol=1e-5, atol=1e-5)

    def test_backward(self) -> None:
        batch_size = 2
        in_channels = 3
        height = 10
        width = 10
        kernel_size = (2, 2)
        stride = (2, 2)

        x_np = np.random.randn(batch_size, in_channels, height, width).astype(
            np.float32
        )

        # Calculate output shape
        out_h = (height - kernel_size[0]) // stride[0] + 1
        out_w = (width - kernel_size[1]) // stride[1] + 1
        grad_output_np = np.random.randn(batch_size, in_channels, out_h, out_w).astype(
            np.float32
        )

        # Custom implementation
        maxpool = MaxPool2d(kernel_size, stride)
        maxpool.forward(x_np)

        dx_custom = maxpool.backward(grad_output_np, x_np)

        # PyTorch implementation
        x_torch = torch.from_numpy(x_np)
        x_torch.requires_grad = True
        grad_output_torch = torch.from_numpy(grad_output_np)

        maxpool_torch = torch.nn.MaxPool2d(kernel_size, stride=stride)
        out_torch = maxpool_torch(x_torch)
        out_torch.backward(grad_output_torch)

        assert x_torch.grad is not None
        np.testing.assert_allclose(
            dx_custom, x_torch.grad.numpy(), rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
