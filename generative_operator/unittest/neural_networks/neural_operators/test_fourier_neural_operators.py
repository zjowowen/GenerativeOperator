import unittest
import torch
from typing import List
import pytest
from generative_operator.neural_networks.neural_operators.fourier_neural_operators import t_allhot, make_posn_embed, FourierNeuralOperator


class TestTAllHot(unittest.TestCase):
    def test_basic(self):
        # Test with a time tensor of shape [batch_size]
        t = torch.tensor([1.0, 2.0])
        output = t_allhot(t, [2, 3, 4])
        self.assertEqual(output.shape, (2, 3, 4))
        
        # Check the per-batch, per-channel expansion
        self.assertTrue(torch.allclose(output[0], torch.ones(3, 4) * 1.0))
        self.assertTrue(torch.allclose(output[1], torch.ones(3, 4) * 2.0))

    def test_scalar_t(self):
        # Test when `t` is just a single scalar (treat as repeated across all batch elements).
        t = torch.tensor(5.0)
        output = t_allhot(t, [2, 2, 2])
        self.assertEqual(output.shape, (2, 2, 2))
        
        # Since t is a single scalar, each element in the output should be 5.0
        self.assertTrue(torch.allclose(output, torch.ones(2, 2, 2) * 5.0))

class TestMakePosnEmbed(unittest.TestCase):
    def test_1d_embeddings(self):
        batch_size = 2
        dims = [5]
        emb = make_posn_embed(batch_size, dims)
        self.assertEqual(emb.shape, (2, 1, 5))
        
    def test_2d_embeddings(self):
        batch_size = 2
        dims = [3, 4]
        emb = make_posn_embed(batch_size, dims)
        self.assertEqual(emb.shape, (2, 2, 3, 4))
        
    def test_3d_embeddings(self):
        batch_size = 2
        dims = [2, 3, 4]
        emb = make_posn_embed(batch_size, dims)
        self.assertEqual(emb.shape, (2, 3, 2, 3, 4))
        
    def test_invalid_dims(self):
        batch_size = 2
        dims = [2, 3, 4, 5]
        with self.assertRaises(NotImplementedError):
            make_posn_embed(batch_size, dims)

    def test_values_range(self):
        """Check if the generated embedding values are within [0, 1]."""
        batch_size = 2
        dims = [3, 4]
        emb = make_posn_embed(batch_size, dims)
        self.assertGreaterEqual(emb.min().item(), 0.0)
        self.assertLessEqual(emb.max().item(), 1.0)

def test_fno_with_condition():
    # Instantiate the model
    model = FourierNeuralOperator(
        modes=4,
        x_channels=3,
        hidden_channels=8,
        proj_channels=8,
        x_dim=2,
        condition_channels=2,
        t_scaling=2.0,
    )

    # Create inputs
    batch_size = 2
    height, width = 32, 32
    t = torch.tensor([1.0, 2.0])  # Time tensor for each item in the batch
    x = torch.randn(batch_size, 3, height, width)
    condition = torch.randn(batch_size, 2, height, width)

    # Forward pass
    output = model(t, x, condition=condition)

    # Assertions
    assert output.shape == (batch_size, 3, height, width), (
        f"Output shape {output.shape} does not match expected "
        f"(batch_size, vis_channels, height, width)."
    )

    # Optional: check for non-nan or non-inf values
    assert torch.isfinite(output).all(), "Output contains invalid values (NaN or Inf)."

def test_fno_without_condition():
    # Instantiate the model
    model = FourierNeuralOperator(
        modes=4,
        x_channels=3,
        hidden_channels=8,
        proj_channels=8,
        x_dim=2,
        condition_channels=None,
        t_scaling=2.0,
    )

    # Create inputs
    batch_size = 2
    height, width = 32, 32
    t = torch.tensor([1.0, 2.0])  # Time tensor for each item in the batch
    x = torch.randn(batch_size, 3, height, width)

    # Forward pass
    output = model(t, x)

    # Assertions
    assert output.shape == (batch_size, 3, height, width), (
        f"Output shape {output.shape} does not match expected "
        f"(batch_size, vis_channels, height, width)."
    )

    # Optional: check for non-nan or non-inf values
    assert torch.isfinite(output).all(), "Output contains invalid values (NaN or Inf)."

if __name__ == '__main__':
    test_fno_without_condition()
    test_fno_with_condition()
    unittest.main()