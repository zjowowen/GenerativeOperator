from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import nn
from torchdiffeq import odeint as torchdiffeq_odeint
from torchdiffeq import odeint_adjoint as torchdiffeq_odeint_adjoint

class ODESolver:
    """
    Overview:
        The ODE solver class.
    Interfaces:
        ``__init__``, ``integrate``
    """

    def __init__(
        self,
        ode_solver="euler",
        dt=0.01,
        atol=1e-5,
        rtol=1e-5,
        library="torchdiffeq",
        **kwargs,
    ):
        """
        Overview:
            Initialize the ODE solver using torchdiffeq library.
        Arguments:
            ode_solver (:obj:`str`): The ODE solver to use.
            dt (:obj:`float`): The time step.
            atol (:obj:`float`): The absolute tolerance.
            rtol (:obj:`float`): The relative tolerance.
            library (:obj:`str`): The library to use for the ODE solver. Currently, it supports 'torchdiffeq'.
            **kwargs: Additional arguments for the ODE solver.
        """
        self.ode_solver = ode_solver
        self.dt = dt
        self.atol = atol
        self.rtol = rtol
        self.nfe = 0
        self.kwargs = kwargs
        self.library = library

    def integrate(
        self,
        drift: Union[nn.Module, Callable],
        x0: Union[torch.Tensor, TensorDict],
        t_span: torch.Tensor,
        **kwargs,
    ):
        """
        Overview:
            Integrate the ODE.
        Arguments:
            drift (:obj:`Union[nn.Module, Callable]`): The drift term of the ODE.
            x0 (:obj:`Union[torch.Tensor, TensorDict]`): The input initial state.
            t_span (:obj:`torch.Tensor`): The time at which to evaluate the ODE. The first element is the initial time, and the last element is the final time. For example, t = torch.tensor([0.0, 1.0]).
        Returns:
            trajectory (:obj:`Union[torch.Tensor, TensorDict]`): The output trajectory of the ODE, which has the same data type as x0 and the shape of (len(t_span), *x0.shape).
        """

        self.nfe = 0
        if self.library == "torchdiffeq":
            return self.odeint_by_torchdiffeq(drift, x0, t_span)
        elif self.library == "torchdiffeq_adjoint":
            return self.odeint_by_torchdiffeq_adjoint(drift, x0, t_span, **kwargs)
        else:
            raise ValueError(f"library {self.library} is not supported")

    def odeint_by_torchdiffeq(self, drift, x0, t_span, **kwargs):
        if isinstance(x0, torch.Tensor):

            def forward_ode_drift_by_torchdiffeq(t, x):
                self.nfe += 1
                # broadcasting t to match the batch size of x
                t = t.repeat(x.shape[0])
                return drift(t, x)

            trajectory = torchdiffeq_odeint(
                func=forward_ode_drift_by_torchdiffeq,
                y0=x0,
                t=t_span,
                method=self.ode_solver,
                atol=self.atol,
                rtol=self.rtol,
                **self.kwargs,
                **kwargs,
            )
            return trajectory

        elif isinstance(x0, Tuple):

            def forward_ode_drift_by_torchdiffeq(t, x):
                self.nfe += 1
                # broadcasting t to match the batch size of x
                t = t.repeat(x[0].shape[0])
                return drift(t, x)

            trajectory = torchdiffeq_odeint(
                func=forward_ode_drift_by_torchdiffeq,
                y0=x0,
                t=t_span,
                method=self.ode_solver,
                atol=self.atol,
                rtol=self.rtol,
                **self.kwargs,
                **kwargs,
            )
            return trajectory

        else:
            raise ValueError(f"Unsupported data type for x0: {type(x0)}")

    def odeint_by_torchdiffeq_adjoint(self, drift, x0, t_span, **kwargs):
        if isinstance(x0, torch.Tensor):

            def forward_ode_drift_by_torchdiffeq_adjoint(t, x):
                self.nfe += 1
                # broadcasting t to match the batch size of x
                t = t.repeat(x.shape[0])
                return drift(t, x)

            trajectory = torchdiffeq_odeint_adjoint(
                func=forward_ode_drift_by_torchdiffeq_adjoint,
                y0=x0,
                t=t_span,
                method=self.ode_solver,
                atol=self.atol,
                rtol=self.rtol,
                **self.kwargs,
                **kwargs,
            )
            return trajectory

        elif isinstance(x0, Tuple):

            def forward_ode_drift_by_torchdiffeq_adjoint(t, x):
                self.nfe += 1
                # broadcasting t to match the batch size of x
                t = t.repeat(x[0].shape[0])
                return drift(t, x)

            trajectory = torchdiffeq_odeint_adjoint(
                func=forward_ode_drift_by_torchdiffeq_adjoint,
                y0=x0,
                t=t_span,
                method=self.ode_solver,
                atol=self.atol,
                rtol=self.rtol,
                **self.kwargs,
                **kwargs,
            )
            return trajectory
