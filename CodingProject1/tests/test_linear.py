import unittest

import numpy as np
import torch

from modules.linear import Linear


class TestLinear(unittest.TestCase):
    def test_forward(self) -> None:
        batch_size = 10
        in_features = 20
        out_features = 5

        x_np = np.random.randn(batch_size, in_features).astype(np.float32)

        # Custom implementation
        linear = Linear(in_features, out_features)
        weight_np = np.random.randn(in_features, out_features).astype(np.float32)
        bias_np = np.random.randn(out_features).astype(np.float32)
        linear.weight = weight_np
        linear.bias = bias_np

        out_custom = linear.forward(x_np)

        # PyTorch implementation
        linear_torch = torch.nn.Linear(in_features, out_features)
        with torch.no_grad():
            linear_torch.weight.copy_(torch.from_numpy(weight_np.T))
            linear_torch.bias.copy_(torch.from_numpy(bias_np))

        x_torch = torch.from_numpy(x_np)
        out_torch = linear_torch(x_torch)

        np.testing.assert_allclose(
            out_custom, out_torch.detach().numpy(), rtol=1e-5, atol=1e-5
        )

    def test_backward(self) -> None:
        batch_size = 10
        in_features = 20
        out_features = 5

        x_np = np.random.randn(batch_size, in_features).astype(np.float32)
        grad_output_np = np.random.randn(batch_size, out_features).astype(np.float32)

        # Custom implementation
        linear = Linear(in_features, out_features)
        weight_np = np.random.randn(in_features, out_features).astype(np.float32)
        bias_np = np.random.randn(out_features).astype(np.float32)
        linear.weight = weight_np
        linear.bias = bias_np

        linear.forward(x_np)

        dx_custom, grads_custom = linear.backward(grad_output_np, x_np)

        # PyTorch implementation
        linear_torch = torch.nn.Linear(in_features, out_features)
        with torch.no_grad():
            linear_torch.weight.copy_(torch.from_numpy(weight_np.T))
            linear_torch.bias.copy_(torch.from_numpy(bias_np))

        x_torch = torch.from_numpy(x_np)
        x_torch.requires_grad = True
        grad_output_torch = torch.from_numpy(grad_output_np)

        out_torch = linear_torch(x_torch)
        out_torch.backward(grad_output_torch)

        assert x_torch.grad is not None
        np.testing.assert_allclose(
            dx_custom, x_torch.grad.numpy(), rtol=1e-5, atol=1e-5
        )

        assert linear_torch.weight.grad is not None
        np.testing.assert_allclose(
            grads_custom.weight,
            linear_torch.weight.grad.numpy().T,
            rtol=1e-5,
            atol=1e-5,
        )

        assert linear_torch.bias.grad is not None
        np.testing.assert_allclose(
            grads_custom.bias, linear_torch.bias.grad.numpy(), rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
