import unittest

import numpy as np
import torch

from modules.activation import ReLU, Tanh


class TestActivation(unittest.TestCase):
    def test_relu_forward(self) -> None:
        x_np = np.random.randn(10, 20).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # Custom implementation
        relu = ReLU()
        out_custom = relu.forward(x_np)

        # PyTorch implementation
        relu_torch = torch.nn.ReLU()
        out_torch = relu_torch(x_torch)

        np.testing.assert_allclose(out_custom, out_torch.numpy(), rtol=1e-5, atol=1e-5)

    def test_relu_backward(self) -> None:
        x_np = np.random.randn(10, 20).astype(np.float32)
        grad_np = np.random.randn(10, 20).astype(np.float32)

        x_torch = torch.from_numpy(x_np)
        x_torch.requires_grad = True
        grad_torch = torch.from_numpy(grad_np)

        # Custom implementation
        relu = ReLU()
        relu.forward(x_np)

        dx_custom = relu.backward(grad_np, x_np)

        # PyTorch implementation
        relu_torch = torch.nn.ReLU()
        out_torch = relu_torch(x_torch)
        out_torch.backward(grad_torch)

        assert x_torch.grad is not None
        np.testing.assert_allclose(
            dx_custom, x_torch.grad.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_tanh_forward(self) -> None:
        x_np = np.random.randn(10, 20).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        tanh = Tanh()
        out_custom = tanh.forward(x_np)

        tanh_torch = torch.nn.Tanh()
        out_torch = tanh_torch(x_torch)

        np.testing.assert_allclose(out_custom, out_torch.numpy(), rtol=1e-5, atol=1e-5)

    def test_tanh_backward(self) -> None:
        x_np = np.random.randn(10, 20).astype(np.float32)
        grad_np = np.random.randn(10, 20).astype(np.float32)

        x_torch = torch.from_numpy(x_np)
        x_torch.requires_grad = True
        grad_torch = torch.from_numpy(grad_np)

        tanh = Tanh()
        tanh.forward(x_np)

        dx_custom = tanh.backward(grad_np, x_np)

        tanh_torch = torch.nn.Tanh()
        out_torch = tanh_torch(x_torch)
        out_torch.backward(grad_torch)

        assert x_torch.grad is not None
        np.testing.assert_allclose(
            dx_custom, x_torch.grad.numpy(), rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
