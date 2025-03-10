from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neuralop.models import FNO


def t_allhot(t: torch.Tensor, shape: List[int]) -> torch.Tensor:
    """
    Overview:
        Expands a scalar or batch-wise time tensor `t` into a tensor of the specified shape.
        The first item in `shape` is assumed to be the batch size, the second is the channel size, 
        and the remaining items define the spatial or additional dimensions.

    Arguments:
        - t (torch.Tensor): A time tensor (can be scalar or have a batch dimension).
        - shape (List[int]): Desired output shape. The first element should match `t.shape[0]`
          if `t` has a batch dimension. For example, [batch_size, n_channels, *dims].

    Returns:
        - torch.Tensor: A tensor of repeated values from `t`, matching the shape [batch_size, n_channels, *dims].

    Example:
        >>> t = torch.tensor([1.0, 2.0])
        >>> output = t_allhot(t, [2, 3, 4])
        >>> output.shape
        torch.Size([2, 3, 4])
    """
    batch_size, n_channels, *dims = shape
    # Reshape `t` to (batch_size, 1, ..., 1)
    #    If `t` is already (batch_size, ) then this adds extra singleton dimensions.
    #    If `t` is a scalar, we treat it as repeated for each batch element.
    if t.dim() == 0 or t.numel() == 1:
        t_reshaped = t.repeat(batch_size, *([1] * (len(dims)+1)))
    else:
        t_reshaped = t.reshape(batch_size, *([1] * (len(dims)+1)))  

    # Create a tensor of ones with the desired final shape
    ones_tensor = torch.ones(batch_size, n_channels, *dims, device=t.device, dtype=t.dtype)

    # Extend the scalar/batch time across n_channels and dims
    return t_reshaped * ones_tensor

def make_posn_embed(batch_size: int, dims: List[int]) -> torch.Tensor:
    """
    Overview:
        Create spatial embeddings for an input grid of up to 3 dimensions.
        For each dimension, a linearly spaced vector in [0, 1] is generated and then repeated
        to match the desired batch size and spatial dimensions.
    
    Arguments:
        - batch_size (int): The size of the batch.
        - dims (List[int]): Dimensions of the grid. Must be of length 1, 2, or 3.
    
    Returns:
        - torch.Tensor: A spatial embedding tensor. Possible shapes:
            1) (batch_size, 1, dims[0]) for 1D,
            2) (batch_size, 2, dims[0], dims[1]) for 2D, and
            3) (batch_size, 3, dims[0], dims[1], dims[2]) for 3D.
    
    Example:
        >>> import torch
        >>> batch_size = 2
        >>> dims = [3, 4]
        >>> emb = make_posn_embed(batch_size, dims)
        >>> emb.shape
        torch.Size([2, 2, 3, 4])
    """
    if len(dims) == 1:
        # 1D embedding
        emb = torch.linspace(0, 1, dims[0]).unsqueeze(0)            # (1, dims[0])
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1)             # (batch_size, 1, dims[0])
    
    elif len(dims) == 2:
        # 2D embedding
        x = torch.linspace(0, 1, dims[1]).repeat(dims[0], 1).unsqueeze(0)   # (1, dims[0], dims[1])
        y = torch.linspace(0, 1, dims[0]).repeat(dims[1], 1).T.unsqueeze(0) # (1, dims[0], dims[1])
        emb = torch.cat((x, y), dim=0)                                      # (2, dims[0], dims[1])
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)      # (batch_size, 2, dims[0], dims[1])
    
    elif len(dims) == 3:
        # 3D embedding
        x = (
            torch.linspace(0, 1, dims[0])
            .reshape(1, dims[0], 1, 1)
            .repeat(1, 1, dims[1], dims[2])
        )
        y = (
            torch.linspace(0, 1, dims[1])
            .reshape(1, 1, dims[1], 1)
            .repeat(1, dims[0], 1, dims[2])
        )
        z = (
            torch.linspace(0, 1, dims[2])
            .reshape(1, 1, 1, dims[2])
            .repeat(1, dims[0], dims[1], 1)
        )
        emb = torch.cat((x, y, z), dim=0)               # (3, dims[0], dims[1], dims[2])
        emb = emb.unsqueeze(0).repeat(
            batch_size, 1, 1, 1, 1
        )                                               # (batch_size, 3, dims[0], dims[1], dims[2])
    
    else:
        raise NotImplementedError("Only 1D, 2D, or 3D embeddings are supported.")
    
    return emb

class FourierNeuralOperator(nn.Module):
    """
    Overview:
        Implements a Fourier Neural Operator model that learns to map input states
        (plus optional conditions and time embeddings) to output states across one
        or more spatial dimensions.

    Interfaces:
        - __init__
        - forward
    """

    def __init__(
        self,
        modes: int,
        x_channels: int,
        hidden_channels: int,
        proj_channels: int,
        x_dim: int = 1,
        condition_channels: int = None,
        t_scaling: float = 1,
    ):
        """
        Overview:
            Initialize the FNO model.

        Arguments:
            - modes (int): Number of Fourier modes.
            - x_channels (int): Number of visual channels.
            - hidden_channels (int): Number of hidden channels in the FNO layers.
            - proj_channels (int): Number of projection channels.
            - x_dim (int): Number of spatial dimensions.
            - condition_channels (int, optional): Number of channels in the conditional tensor.
            - t_scaling (float): Scaling factor applied to the input time.
        """
        super().__init__()

        self.t_scaling = t_scaling
        n_modes = (modes,) * x_dim  # Same number of modes for each spatial dimension

        # in_channels = x_channels + x_dim + 1 + condition_channels(optional):
        #   - x_channels: data channels
        #   - x_dim: positional embeddings
        #   - 1: scaled time embedding
        #   - condition_channels: additional conditioning tensor
        if condition_channels is not None:
            in_channels = x_channels + x_dim + 1 + condition_channels
        else:
            in_channels = x_channels + x_dim + 1

        self.model = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            projection_channels=proj_channels,
            in_channels=in_channels,
            out_channels=x_channels,
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Overview:
            Forward pass of the FNO model. Scales the time input, creates a positional
            embedding, optionally concatenates a conditional tensor, and produces an output.

        Arguments:
            - t (torch.Tensor): Time tensor, which can be a scalar or [batch_size].
            - x (torch.Tensor): Input tensor with shape [batch_size, channels, *spatial_dims].
            - condition (torch.Tensor, optional): An additional conditioning tensor to concatenate.

        Returns:
            - out (torch.Tensor): Output tensor with shape [batch_size, vis_channels, *spatial_dims].
        """
        # Scale the time
        t = t / self.t_scaling

        batch_size = x.shape[0]
        dims = x.shape[2:]  # Remaining spatial dimensions

        # Expand time tensor to batch dimension if it's a single scalar
        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(batch_size, device=x.device) * t

        assert t.dim() == 1 and t.shape[0] == batch_size, (
            "The time tensor must match the batch dimension."
        )

        # Create a time embedding matching [batch_size, 1, *spatial_dims]
        t_expanded = t_allhot(t, [batch_size, 1, *dims])

        # Create positional embedding
        posn_emb = make_posn_embed(batch_size, dims).to(x.device)

        # Concatenate along the channel dimension
        if condition is not None:
            u = torch.cat([x, posn_emb, t_expanded, condition], dim=1).float()
        else:
            u = torch.cat([x, posn_emb, t_expanded], dim=1).float()

        out = self.model(u)
        return out
