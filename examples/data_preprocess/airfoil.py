import os
from typing import List
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from tensordict import TensorDict

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed
from easydict import EasyDict
from easydict import EasyDict

from generative_operator.model.point_cloud_flow_model import PointCloudFunctionalFlow
from generative_operator.dataset.tensordict_dataset import TensorDictDataset

from generative_operator.neural_networks.neural_operators.point_cloud_neural_operator import (
    preprocess_data,
    compute_node_weights,
    compute_Fourier_modes,
)
from generative_operator.neural_networks.neural_operators.point_cloud_data_process import (
    convert_structured_data,
)


from generative_operator.utils.normalizer import UnitGaussianNormalizer


def data_preprocess(
    data_path, file_name="pcno_quad_data.npz", value_type: List[str] = ["mach"]
):
    """
    Overview:
        Preprocess the data for the PCNO model.
    Arguments:
        - data_path (str): path to the data
        - file_name (str): name of the output file
        - value_type (List[str]): type of the data to be used
        .. note::
            The value_type can be one of the following:
            - "density"
            - "velocityx"
            - "velocityy"
            - "pressure"
            - "mach"
            The default value is ["mach"].
    """

    value_type_idx = []
    for value in value_type:
        if value == "density":
            value_type_idx.append(0)
        elif value == "velocityx":
            value_type_idx.append(1)
        elif value == "velocityy":
            value_type_idx.append(2)
        elif value == "pressure":
            value_type_idx.append(3)
        elif value == "mach":
            value_type_idx.append(4)
        else:
            raise ValueError(
                f"Invalid value type: {value}. Must be one of ['density', 'velocity', 'pressure', 'mach']"
            )

    coordx = np.load(os.path.join(data_path, "NACA_Cylinder_X.npy"))
    coordy = np.load(os.path.join(data_path, "NACA_Cylinder_Y.npy"))
    data_out = np.load(os.path.join(data_path, "NACA_Cylinder_Q.npy"))[
        :, value_type_idx, :, :
    ]

    nodes_list, elems_list, features_list = convert_structured_data(
        [coordx, coordy],
        data_out[..., np.newaxis],
        nnodes_per_elem=4,
        feature_include_coords=False,
    )

    (
        nnodes,
        node_mask,
        nodes,
        node_measures_raw,
        features,
        directed_edges,
        edge_gradient_weights,
    ) = preprocess_data(nodes_list, elems_list, features_list)
    node_measures, node_weights = compute_node_weights(
        nnodes, node_measures_raw, equal_measure=False
    )
    node_equal_measures, node_equal_weights = compute_node_weights(
        nnodes, node_measures_raw, equal_measure=True
    )
    np.savez_compressed(
        os.path.join(data_path, file_name),
        nnodes=nnodes,
        node_mask=node_mask,
        nodes=nodes,
        node_measures_raw=node_measures_raw,
        node_measures=node_measures,
        node_weights=node_weights,
        node_equal_measures=node_equal_measures,
        node_equal_weights=node_equal_weights,
        features=features,
        directed_edges=directed_edges,
        edge_gradient_weights=edge_gradient_weights,
    )


def load_data(data_path, file_name="pcno_quad_data.npz"):
    equal_weights = True

    data = np.load(os.path.join(data_path, file_name))
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
    node_measures = data["node_measures"]
    directed_edges, edge_gradient_weights = (
        data["directed_edges"],
        data["edge_gradient_weights"],
    )
    features = data["features"]

    node_measures_raw = data["node_measures_raw"]
    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices] / node_measures[indices]
    return (
        nnodes,
        node_mask,
        nodes,
        node_weights,
        node_rhos,
        features,
        directed_edges,
        edge_gradient_weights,
    )


def data_preparition_with_tensordict(
    nnodes,
    node_mask,
    nodes,
    node_weights,
    node_rhos,
    features,
    directed_edges,
    edge_gradient_weights,
    n_train=1000,
    n_test=200,
    normalization_x=False,
    normalization_y=False,
    normalization_dim_x=[],
    normalization_dim_y=[],
    non_normalized_dim_x=3,
    non_normalized_dim_y=0,
):
    print("Casting to tensor")
    nnodes = torch.from_numpy(nnodes)
    node_mask = torch.from_numpy(node_mask)
    nodes = torch.from_numpy(nodes.astype(np.float32))
    node_weights = torch.from_numpy(node_weights.astype(np.float32))
    node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
    features = torch.from_numpy(features.astype(np.float32))
    directed_edges = torch.from_numpy(directed_edges)
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

    # This is important
    nodes_input = nodes.clone()

    train_data = TensorDict(
        {
            "y": features[:n_train, ...],
            "condition": TensorDict(
                {
                    "x": torch.cat(
                        (nodes_input[:n_train, ...], node_rhos[:n_train, ...]), -1
                    ),
                    "node_mask": node_mask[0:n_train, ...],
                    "nodes": nodes[0:n_train, ...],
                    "node_weights": node_weights[0:n_train, ...],
                    "directed_edges": directed_edges[0:n_train, ...],
                    "edge_gradient_weights": edge_gradient_weights[0:n_train, ...],
                },
                batch_size=(n_train,),
            ),
        },
        batch_size=(n_train,),
    )
    test_data = TensorDict(
        {
            "y": features[-n_test:, ...],
            "condition": TensorDict(
                {
                    "x": torch.cat(
                        (nodes_input[-n_test:, ...], node_rhos[-n_test:, ...]), -1
                    ),
                    "node_mask": node_mask[-n_test:, ...],
                    "nodes": nodes[-n_test:, ...],
                    "node_weights": node_weights[-n_test:, ...],
                    "directed_edges": directed_edges[-n_test:, ...],
                    "edge_gradient_weights": edge_gradient_weights[-n_test:, ...],
                },
                batch_size=(n_test,),
            ),
        },
        batch_size=(n_test,),
    )

    n_train, n_test = (
        train_data["condition"]["x"].shape[0],
        test_data["condition"]["x"].shape[0],
    )

    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(
            train_data["x"],
            non_normalized_dim=non_normalized_dim_x,
            normalization_dim=normalization_dim_x,
        )
        x_normalizer.to(device)
    else:
        x_normalizer = None

    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(
            train_data["y"],
            non_normalized_dim=non_normalized_dim_y,
            normalization_dim=normalization_dim_y,
        )
        y_normalizer.to(device)
    else:
        y_normalizer = None

    train_dataset = TensorDictDataset(keys=["y", "condition"], max_size=n_train)
    test_dataset = TensorDictDataset(keys=["y", "condition"], max_size=n_test)
    train_dataset.append(train_data, batch_size=n_train)
    test_dataset.append(test_data, batch_size=n_test)

    return train_dataset, test_dataset, x_normalizer, y_normalizer


def model_initialization(
    device, x_train, y_train, kx_max=32, ky_max=16, Lx=4.0, Ly=4.0, ndims=2
):
    modes = compute_Fourier_modes(ndims, [kx_max, ky_max], [Lx, Ly])
    modes = torch.tensor(modes, dtype=torch.float).to(device)

    flow_model_config = EasyDict(
        dict(
            device=device,
            gaussian_process=dict(
                type="Matern",
                args=dict(
                    length_scale=0.01,
                    nu=1.5,
                ),
            ),
            solver=dict(
                type="ODESolver",
                args=dict(
                    library="torchdiffeq",
                ),
            ),
            path=dict(
                sigma=1e-4,
                device=device,
            ),
            model=dict(
                type="velocity_function",
                args=dict(
                    backbone=dict(
                        type="PointCloudNeuralOperator",
                        args=dict(
                            ndims=ndims,
                            modes=modes,
                            nmeasures=1,
                            layers=[128, 128, 128, 128, 128],
                            fc_dim=128,
                            in_dim=y_train.shape[-1] + 1 + x_train.shape[-1],
                            out_dim=y_train.shape[-1],
                            train_sp_L="independently",
                            act="gelu",
                        ),
                    ),
                ),
            ),
        ),
    )

    model = PointCloudFunctionalFlow(
        config=flow_model_config,
    )

    return model, flow_model_config


if __name__ == "__main__":
    # argparse project_name
    import argparse

    parser = argparse.ArgumentParser(description="PCNO flow model training")
    parser.add_argument(
        "--project_name",
        type=str,
        default="PCNO_airfoil",
        help="Project name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/d/Dataset/",  # "data/",
        help="Path to the data",
    )
    parser.add_argument(
        "--value_type",
        type=lambda s: s.split(","),
        default=["mach"],
        help="Type of the data to be used, separated by comma. Options: density, velocityx, velocityy, pressure, mach",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="pcno_quad_data.npz",
        help="Name of the output file",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use wandb",
    )
    args = parser.parse_args()
    project_name = args.project_name
    seed = args.seed
    value_type = args.value_type
    file_name = args.file_name

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="wandb" if args.wandb else None, kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device
    state = AcceleratorState()

    # Get the process rank
    process_rank = state.process_index
    set_seed(seed=seed + process_rank)
    print(f"Process rank: {process_rank}")

    # check GPU brand, if NVIDIA RTX 4090 use batch size 4, if NVIDIA A100 use batch size 16
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "A100" in gpu_name or "A800" in gpu_name:
            batch_size = 16 // len(value_type)
        elif "4090" in gpu_name:
            batch_size = 4 // len(value_type)
        elif "H200" in gpu_name:
            batch_size = 32 // len(value_type)
        else:
            batch_size = 4 // len(value_type)
        print(f"GPU name: {gpu_name}, batch size: {batch_size}")
    else:
        gpu_name = "CPU"
        batch_size = 4 // len(value_type)
        print(f"CPU, batch size: {batch_size}")

    if args.data_preprocess:
        data_preprocess(args.data_path, file_name=file_name, value_type=value_type)
    data_path = args.data_path

    print("data preprocess finished")

    (
        nnodes,
        node_mask,
        nodes,
        node_weights,
        node_rhos,
        features,
        directed_edges,
        edge_gradient_weights,
    ) = load_data(data_path, file_name=file_name)

    print("Data loaded")