import unittest

import numpy as np
import torch

from modules.conv import Conv2d


class TestConv2d(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.in_channels = 3
        self.out_channels = 4
        self.height = 10
        self.width = 10
        self.kernel_size = (3, 3)
        self.stride = (1, 1)

    def test_forward(self) -> None:
        x_np = np.random.randn(
            self.batch_size, self.in_channels, self.height, self.width
        ).astype(np.float32)

        # Custom implementation
        conv = Conv2d(
            self.in_channels, self.out_channels, self.kernel_size, self.stride
        )
        weight_np = np.random.randn(
            self.out_channels, self.in_channels, *self.kernel_size
        ).astype(np.float32)
        bias_np = np.random.randn(self.out_channels).astype(np.float32)
        conv.weight = weight_np
        conv.bias = bias_np

        out_custom = conv.forward(x_np)

        # PyTorch implementation
        conv_torch = torch.nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size, stride=self.stride
        )
        assert conv_torch.bias is not None
        with torch.no_grad():
            conv_torch.weight.copy_(torch.from_numpy(weight_np))
            conv_torch.bias.copy_(torch.from_numpy(bias_np))

        x_torch = torch.from_numpy(x_np)
        out_torch = conv_torch(x_torch)

        np.testing.assert_allclose(
            out_custom, out_torch.detach().numpy(), rtol=1e-4, atol=1e-4
        )

    def test_backward(self) -> None:
        x_np = np.random.randn(
            self.batch_size, self.in_channels, self.height, self.width
        ).astype(np.float32)
        weight_np = np.random.randn(
            self.out_channels, self.in_channels, *self.kernel_size
        ).astype(np.float32)
        bias_np = np.random.randn(self.out_channels).astype(np.float32)

        h_out = (self.height - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (self.width - self.kernel_size[1]) // self.stride[1] + 1
        grad_output_np = np.random.randn(
            self.batch_size, self.out_channels, h_out, w_out
        ).astype(np.float32)

        # Custom implementation
        conv = Conv2d(
            self.in_channels, self.out_channels, self.kernel_size, self.stride
        )
        conv.weight = weight_np
        conv.bias = bias_np

        conv.forward(x_np)

        dx_custom, grads_custom = conv.backward(grad_output_np, x_np)

        # PyTorch implementation
        conv_torch = torch.nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size, stride=self.stride
        )
        assert conv_torch.bias is not None
        with torch.no_grad():
            conv_torch.weight.copy_(torch.from_numpy(weight_np))
            conv_torch.bias.copy_(torch.from_numpy(bias_np))

        x_torch = torch.from_numpy(x_np)
        x_torch.requires_grad = True
        grad_output_torch = torch.from_numpy(grad_output_np)

        out_torch = conv_torch(x_torch)
        out_torch.backward(grad_output_torch)

        assert x_torch.grad is not None
        np.testing.assert_allclose(
            dx_custom, x_torch.grad.numpy(), rtol=1e-4, atol=1e-4
        )

        assert conv_torch.weight.grad is not None
        np.testing.assert_allclose(
            grads_custom.weight, conv_torch.weight.grad.numpy(), rtol=1e-4, atol=1e-4
        )

        assert conv_torch.bias.grad is not None
        np.testing.assert_allclose(
            grads_custom.bias, conv_torch.bias.grad.numpy(), rtol=1e-4, atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
