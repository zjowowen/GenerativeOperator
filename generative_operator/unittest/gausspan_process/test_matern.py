import unittest
import numpy as np
import torch
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

from generative_operator.gaussian_process.matern import (
    matern_halfinteger_kernel,
    MaternGaussianProcess,
)

class TestMaternGaussianProcess(unittest.TestCase):
    def setUp(self):
        # Set seed for reproducibility
        self.seed = 42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # Generate test data
        self.N = 5
        self.D = 2
        self.X_torch = torch.randn(self.N, self.D, dtype=torch.float64)
        self.X_np = self.X_torch.numpy()

    def _test_matern_kernel(self, nu, lengthscale, variance):
        # Compute custom kernel
        K_custom = matern_halfinteger_kernel(
            self.X_torch, self.X_torch, lengthscale, nu, variance
        ).numpy()

        # Compute sklearn kernel
        sklearn_kernel = ConstantKernel(constant_value=variance) * Matern(
            length_scale=lengthscale, nu=nu
        )
        K_sklearn = sklearn_kernel(self.X_np)

        # Check closeness
        np.testing.assert_allclose(K_custom, K_sklearn, rtol=1e-5, atol=1e-8)

        # Check GP class covariance matrix
        gp = MaternGaussianProcess(
            self.X_torch,
            lengthscale=lengthscale,
            nu=nu,
            variance=variance
        )
        K_gp = gp._cov_matrix.numpy()
        np.testing.assert_allclose(K_gp, K_sklearn, rtol=1e-5, atol=1e-8)

    def test_nu_0_5(self):
        self._test_matern_kernel(nu=0.5, lengthscale=1.0, variance=1.0)

    def test_nu_1_5(self):
        self._test_matern_kernel(nu=1.5, lengthscale=0.5, variance=2.0)

    def test_nu_2_5(self):
        self._test_matern_kernel(nu=2.5, lengthscale=2.0, variance=0.5)

    def test_nu_3_5(self):
        self._test_matern_kernel(nu=3.5, lengthscale=1.5, variance=3.0)

    def test_invalid_nu_raises_error(self):
        X = torch.randn(3, 2, dtype=torch.float64)
        gp = MaternGaussianProcess(X, lengthscale=1.0, nu=2.0, variance=1.0)
        with self.assertRaises(NotImplementedError):
            _ = gp._cov_matrix

    def test_negative_lengthscale_raises_error(self):
        X = torch.randn(3, 2, dtype=torch.float64)
        gp = MaternGaussianProcess(X, lengthscale=-1.0, nu=1.5, variance=1.0)
        with self.assertRaises(ValueError):
            _ = gp._cov_matrix

    def test_identical_points_covariance(self):
        X = torch.zeros(3, 2, dtype=torch.float64)
        variance = 2.0
        gp = MaternGaussianProcess(X, lengthscale=1.0, nu=1.5, variance=variance)
        K = gp._cov_matrix.numpy()
        expected = np.full((3, 3), variance)
        np.testing.assert_allclose(K, expected, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()