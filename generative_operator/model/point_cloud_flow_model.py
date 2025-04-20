from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import ot
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from easydict import EasyDict

from generative_operator.model.probability_path import ConditionalProbabilityPath
from generative_operator.model.intrinsic_model import IntrinsicModel
from generative_operator.model.stochastic_process import StochasticProcess
from generative_operator.gaussian_process.matern import MaternGaussianProcess
from generative_operator.numerical_solvers import ODESolver
from generative_operator.numerical_solvers import get_solver
from generative_operator.utils import find_parameters
from generative_operator.utils.loss import LpLoss

class PointCloudFunctionalFlow(nn.Module):
    """
    Overview:
        A functional flow model for generative modeling of point cloud data.
    Interface:
        "__init__", "get_type", "forward", "sample", "sample_process", "inverse_sample", "inverse_sample_process", "inverse_sample_with_log_prob", "inverse_sample_process_with_log_prob", "functional_flow_matching_loss"
    """

    def __init__(
        self,
        config: EasyDict,
        model: nn.Module = None,
    ):
        """
        Overview:
            Initialize the functional flow model.
        Arguments:
            - config (EasyDict): configuration for the model
            - model (nn.Module): intrinsic model
        """
        super().__init__()

        self.config = config
        self.device = config.device
        self.path = ConditionalProbabilityPath(config.path)
        self.model = IntrinsicModel(config.model.args) if model is None else model

        def get_gaussian_process_type(type: str, **args) -> Callable:
            if type.lower() == "matern":
                def initialize_matern_gaussian_process(X):
                    return MaternGaussianProcess(
                        X=X,
                        length_scale=args["length_scale"],
                        nu=args["nu"],
                    )
                return initialize_matern_gaussian_process
            else:
                raise ValueError(f"Unknown Gaussian process type: {type}")

        self.gaussian_process = get_gaussian_process_type(
            config.gaussian_process.type,
            **config.gaussian_process.args,
        )

        self.stochastic_process = StochasticProcess(self.path, self.gaussian_process)

        if hasattr(config, "solver"):
            self.solver = get_solver(config.solver.type)(**config.solver.args)

        self.loss_function = LpLoss(d=1, p=2, size_average=True)


    def get_type(self):
        return "PointCloudFunctionalFlow"

    def forward(
        self,
        x,
    ):
        pass

    def sample(
        self,
        x0: torch.Tensor,
        t_span: torch.Tensor = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        condition: torch.Tensor = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
    ):
        """
        Overview:
            Sample from the functional flow model and return the last sample.
        Arguments:
            - x0 (tensor): initial condition for sampling
            - t_span (tensor): time span to sample over
            - batch_size (int, tuple, list): batch size for sampling
            - condition (tensor): condition for sampling
            - with_grad (bool): whether to compute gradients
            - solver_config (EasyDict): configuration for the solver
        Returns:
            - samples (tensor): samples from the functional flow model; tensor (T, B, N, D)
        """
        return self.sample_process(
            x0=x0,
            t_span=t_span,
            batch_size=batch_size,
            condition=condition,
            with_grad=with_grad,
            solver_config=solver_config,
        )[-1]

    def sample_process(
        self,
        x0: torch.Tensor,
        t_span: torch.Tensor = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        condition: torch.Tensor = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
    ):
        """
        Overview:
            Sample from the functional flow model.
        Arguments:
            - x0 (tensor): initial condition for sampling
            - t_span (tensor): time span to sample over
            - batch_size (int, tuple, list): batch size for sampling
            - condition (tensor): condition for sampling
            - with_grad (bool): whether to compute gradients
            - solver_config (EasyDict): configuration for the solver
        Returns:
            - samples (tensor): samples from the functional flow model; tensor (T, B, N, D)
        """
        if t_span is not None:
            t_span = t_span.to(self.device)

        if batch_size is None:
            extra_batch_size = torch.tensor((1,), device=self.device)
        elif isinstance(batch_size, int):
            extra_batch_size = torch.tensor((batch_size,), device=self.device)
        else:
            if (
                isinstance(batch_size, torch.Size)
                or isinstance(batch_size, Tuple)
                or isinstance(batch_size, List)
            ):
                extra_batch_size = torch.tensor(batch_size, device=self.device)
            else:
                assert False, "Invalid batch size"

        if condition is not None:
            assert (
                x0.shape[0] == condition.shape[0]
            ), "The batch size of x0 and condition must be the same"
            data_batch_size = x0.shape[0]
        else:
            data_batch_size = x0.shape[0]


        if solver_config is not None:
            solver = get_solver(solver_config.type)(**solver_config.args)
        else:
            assert hasattr(
                self, "solver"
            ), "solver must be specified in config or solver_config"
            solver = self.solver

        x = x0

        if isinstance(solver, ODESolver):
            # TODO: make it compatible with TensorDict
            def drift(t, x):
                return self.model(t=t, x=x, condition=condition)

            if solver.library == "torchdiffeq_adjoint":
                if with_grad:
                    data = solver.integrate(
                        drift=drift,
                        x0=x,
                        t_span=t_span,
                        adjoint_params=find_parameters(self.model),
                    )
                else:
                    with torch.no_grad():
                        data = solver.integrate(
                            drift=drift,
                            x0=x,
                            t_span=t_span,
                            adjoint_params=find_parameters(self.model),
                        )
            else:
                if with_grad:
                    data = solver.integrate(
                        drift=drift,
                        x0=x,
                        t_span=t_span,
                    )
                else:
                    with torch.no_grad():
                        data = solver.integrate(
                            drift=drift,
                            x0=x,
                            t_span=t_span,
                        )
        else:
            raise NotImplementedError("Not implemented")

        if batch_size is None:
            if x0 is None and condition is None:
                data = data.squeeze(1).squeeze(1)
                # data.shape = (T, D)
            else:
                data = data.squeeze(1)
                # data.shape = (T, N, D)
        else:
            if x0 is None and condition is None:
                data = data.squeeze(1 + len(extra_batch_size.shape))
                # data.shape = (T, B, D)
            else:
                # data.shape = (T, B, N, D)
                pass

        return data



    def functional_flow_matching_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        gaussian_process_samples: float,
        condition: torch.Tensor = None,
        mse_loss: bool = False,
    ):
        """
        Overview:
            Compute the functional flow matching loss.
        Arguments:
            - x0 (tensor): initial condition for sampling
            - x1 (tensor): target condition for sampling
            - gaussian_process_samples (float): samples from the Gaussian process
            - condition (tensor): condition for sampling
        Returns:
            - loss (tensor): functional flow matching loss
        """

        def get_loss(velocity_value, velocity):
            return torch.mean(
                torch.sum(0.5 * (velocity_value - velocity) ** 2, dim=(1,))
            )

        batch_size = x0.shape[0]
        t_random = (
            torch.rand(batch_size, device=self.device) * self.stochastic_process.t_max
        )
        x_t = self.stochastic_process.direct_sample(t_random, x0, x1, gaussian_process_samples)

        velocity_value = self.model(t_random, x_t, condition=condition)
        velocity = self.stochastic_process.velocity(t_random, x0, x1)

        velocity_value_masked = velocity_value * condition["node_mask"]
        velocity_masked = velocity * condition["node_mask"]

        if mse_loss:
            loss = get_loss(velocity_value_masked, velocity_masked)
        else:
            loss = self.loss_function.abs(velocity_value_masked, velocity_masked)
        return loss

