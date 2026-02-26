import unittest

import numpy as np
import torch

from modules.flatten import Unflatten


class TestUnflatten(unittest.TestCase):
    def test_forward(self) -> None:
        x_np = np.random.randn(10, 20).astype(np.float32)
        dim = 1
        unflattened_size = (4, 5)

        # Custom implementation
        unflatten = Unflatten(dim, unflattened_size)
        out_custom = unflatten.forward(x_np)

        # PyTorch implementation
        x_torch = torch.from_numpy(x_np)
        unflatten_torch = torch.nn.Unflatten(dim, unflattened_size)
        out_torch = unflatten_torch(x_torch)

        np.testing.assert_allclose(out_custom, out_torch.numpy(), rtol=1e-5, atol=1e-5)

    def test_backward(self) -> None:
        x_np = np.random.randn(10, 20).astype(np.float32)
        dim = 1
        unflattened_size = (4, 5)

        # Custom implementation
        unflatten = Unflatten(dim, unflattened_size)
        unflatten.forward(x_np)

        grad_output_np = np.random.randn(10, 4, 5).astype(np.float32)

        dx_custom = unflatten.backward(grad_output_np, x_np)

        # PyTorch implementation
        x_torch = torch.from_numpy(x_np)
        x_torch.requires_grad = True
        grad_output_torch = torch.from_numpy(grad_output_np)

        unflatten_torch = torch.nn.Unflatten(dim, unflattened_size)
        out_torch = unflatten_torch(x_torch)
        out_torch.backward(grad_output_torch)

        assert x_torch.grad is not None
        np.testing.assert_allclose(
            dx_custom, x_torch.grad.numpy(), rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
