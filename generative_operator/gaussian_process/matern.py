import math
import torch
import torch.nn as nn
from torch.distributions import Distribution, MultivariateNormal
from torch.distributions.utils import lazy_property


def matern_halfinteger_kernel(X1: torch.Tensor,
                              X2: torch.Tensor,
                              lengthscale: float,
                              nu: float,
                              variance: float
                             ) -> torch.Tensor:
    """
    Computes the Matern kernel for half-integer ν between two sets of inputs X1 and X2.
    
    This function only supports ν in {0.5, 1.5, 2.5, 3.5}. For other values of ν,
    you'll need a more general approach (e.g., using torch.special.kv) or adding
    additional polynomial expansions.

    Args:
        X1 (torch.Tensor):
            Shape (N, D), representing N points in D dimensions.
        X2 (torch.Tensor):
            Shape (M, D).
        lengthscale (float):
            The lengthscale parameter (sometimes denoted ℓ), must be > 0.
        nu (float):
            The half-integer smoothness parameter: {0.5, 1.5, 2.5, 3.5}.
        variance (float):
            The overall variance (sometimes denoted σ²).

    Returns:
        torch.Tensor:
            The (N, M) covariance matrix K(X1, X2).
    """
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be positive.")

    # Compute pairwise distances
    dists = torch.cdist(X1, X2, p=2)  # [N, M]

    # For very small distances, we can clamp them to avoid numerical issues.
    dists_clamped = dists.clamp_min(1e-12)

    # Depending on nu, choose the closed-form expression:
    # ---------------------------------------------------
    # nu = 0.5:
    #   K(r) = variance * exp(-r / ℓ)
    #
    # nu = 1.5:
    #   K(r) = variance * (1 + sqrt(3)*r / ℓ) * exp(-sqrt(3)*r / ℓ)
    #
    # nu = 2.5:
    #   K(r) = variance * (1 + sqrt(5)*r/ℓ + 5*r^2 / (3 ℓ^2)) * exp(-sqrt(5)*r/ℓ)
    #
    # nu = 3.5:
    #   K(r) = variance * (
    #       1 + sqrt(7)*r/ℓ + 14*r^2/(5ℓ^2) + 7*sqrt(7)*r^3/(15ℓ^3)
    #   ) * exp(-sqrt(7)*r/ℓ)

    if nu == 0.5:
        # Exponential kernel
        out = variance * torch.exp(-dists_clamped / lengthscale)
    elif nu == 1.5:
        c = math.sqrt(3)
        term = (1.0 + c * dists_clamped / lengthscale)
        out = variance * term * torch.exp(-c * dists_clamped / lengthscale)
    elif nu == 2.5:
        c = math.sqrt(5)
        r = dists_clamped
        ell = lengthscale
        term = (1.0 + c * r / ell + 5.0 * r.pow(2) / (3.0 * ell**2))
        out = variance * term * torch.exp(-c * r / ell)
    elif nu == 3.5:
        c = math.sqrt(7)
        r = dists_clamped
        ell = lengthscale
        term = (
            1.0 +
            c * r / ell +
            14.0 * r.pow(2) / (5.0 * ell**2) +
            7.0 * c * r.pow(3) / (15.0 * ell**3)
        )
        out = variance * term * torch.exp(-c * r / ell)
    else:
        raise NotImplementedError(
            f"Matern kernel for nu={nu} is not implemented. "
            "Only half-integers {0.5, 1.5, 2.5, 3.5} are supported here."
        )

    # Where distance == 0, the kernel should give variance
    out = torch.where(dists == 0, variance, out)

    return out


class MaternGaussianProcess(Distribution):
    """
    A Gaussian Process with a Matern kernel for half-integer ν values.
    
    This implementation uses a zero mean function and
    torch.distributions.MultivariateNormal internally for sampling and log probability.
    """

    arg_constraints = {}

    def __init__(
        self,
        X: torch.Tensor,
        lengthscale: float = 1.0,
        nu: float = 1.5,
        variance: float = 1.0,
        validate_args=None
    ):
        """
        Args:
            X (torch.Tensor):
                Input data of shape (N, D).
            lengthscale (float):
                Lengthscale parameter ℓ > 0.
            nu (float):
                Half-integer smoothness parameter {0.5, 1.5, 2.5, 3.5}.
            variance (float):
                Variance parameter σ² > 0.
            validate_args:
                Whether to validate distribution arguments.
        """
        super().__init__(validate_args=validate_args)
        self.X = X
        self.lengthscale = lengthscale
        self.nu = nu
        self.variance_ = variance

        self._num_points = X.shape[0]

    @lazy_property
    def _cov_matrix(self) -> torch.Tensor:
        """Compute and cache the Matern covariance matrix K(X, X)."""
        return matern_halfinteger_kernel(
            self.X,
            self.X,
            lengthscale=self.lengthscale,
            nu=self.nu,
            variance=self.variance_
        )

    def expand(self, batch_shape, _instance=None):
        # Not implementing batch shapes for simplicity
        raise NotImplementedError("Batching not implemented for MaternGaussianProcess.")

    @property
    def batch_shape(self):
        # This GP has no explicit batching dimension in this example
        return torch.Size([])

    @property
    def event_shape(self):
        # The event is a vector of length N
        return torch.Size([self._num_points])

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """
        Draw samples from the GP prior. Returns a tensor of shape:
        sample_shape x N
        """
        cov = self._cov_matrix
        mvn = MultivariateNormal(loc=torch.zeros(self._num_points), covariance_matrix=cov)
        return mvn.sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes log probability of observing `value` under the GP prior.
        `value` has shape: sample_shape x N
        """
        cov = self._cov_matrix
        mvn = MultivariateNormal(loc=torch.zeros(self._num_points), covariance_matrix=cov)
        return mvn.log_prob(value)

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """
        Draw a reparameterized sample from the GP.
        """
        cov = self._cov_matrix
        mvn = MultivariateNormal(loc=torch.zeros(self._num_points), covariance_matrix=cov)
        return mvn.rsample(sample_shape)


if __name__ == "__main__":
    # Example usage
    N = 5
    D = 2
    # Some random input data
    X = torch.randn(N, D)

    # Create a Matern GP with half-integer ν
    mgp = MaternGaussianProcess(X, lengthscale=1.0, nu=1.5, variance=2.0)

    # Sample from the GP prior
    samples = mgp.sample(sample_shape=(3,))  # 3 samples, each of dimension N
    print("Samples shape:", samples.shape)   # (3, N)

    # Log probability of a new observation (of shape (N,))
    y = torch.randn(N)
    logp = mgp.log_prob(y)
    print("Log prob:", logp)