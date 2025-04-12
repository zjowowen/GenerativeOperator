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
                        device=self.device,
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

        self.loss_function = LpLoss(d=1, p=2, size_average=False)


    def get_type(self):
        return "PointCloudFunctionalFlow"

    def forward(
        self,
        x,
    ):
        pass


    def functional_flow_matching_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        condition: torch.Tensor = None,
    ):
        """
        Overview:
            Compute the functional flow matching loss.
        Arguments:
            - x0 (tensor): initial condition for sampling
            - x1 (tensor): target condition for sampling
            - condition (tensor): condition for sampling
            - average (bool): whether to average the loss
            - sum_all_elements (bool): whether to sum all elements of the loss
        Returns:
            - loss (tensor): functional flow matching loss
        """

        batch_size = x0.shape[0]
        t_random = (
            torch.rand(batch_size, device=self.device) * self.stochastic_process.t_max
        )
        x_t = self.stochastic_process.direct_sample(t_random, x0, x1)

        velocity_value = self.model(t_random, x_t, condition=condition)
        velocity = self.stochastic_process.velocity(t_random, x0, x1)

        velocity_value_masked = velocity_value * condition["aux"]["node_mask"]
        velocity_masked = velocity * condition["aux"]["node_mask"]

        loss = self.loss_function(velocity_value_masked, velocity_masked)
        return loss

