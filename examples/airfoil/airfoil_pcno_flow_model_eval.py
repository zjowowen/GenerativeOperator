import datetime
import argparse

import os
import torch.multiprocessing as mp


import matplotlib

# matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data import SliceSamplerWithoutReplacement, SliceSampler, RandomSampler

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed

from timeit import default_timer
from easydict import EasyDict

import wandb
from rich.progress import track
from easydict import EasyDict

from generative_operator.model.flow_model import (
    FunctionalFlow,
)
from generative_operator.model.point_cloud_flow_model import PointCloudFunctionalFlow
from generative_operator.utils.optimizer import CosineAnnealingWarmupLR
from generative_operator.dataset.tensordict_dataset import TensorDictDataset

from generative_operator.neural_networks.neural_operators.point_cloud_neural_operator import preprocess_data, compute_node_weights, compute_Fourier_modes
from generative_operator.neural_networks.neural_operators.point_cloud_data_process import compute_triangle_area_, compute_tetrahedron_volume_, compute_measure_per_elem_, compute_node_measures, convert_structured_data

from generative_operator.gaussian_process.matern import matern_halfinteger_kernel_batchwise
from generative_operator.utils.normalizer import UnitGaussianNormalizer


def data_preprocess(data_path):
    coordx    = np.load(data_path+"NACA_Cylinder_X.npy")
    coordy    = np.load(data_path+"NACA_Cylinder_Y.npy")
    data_out  = np.load(data_path+"NACA_Cylinder_Q.npy")[:,4,:,:] #density, velocity 2d, pressure, mach number

    nodes_list, elems_list, features_list = convert_structured_data([coordx, coordy], data_out[...,np.newaxis], nnodes_per_elem = 4, feature_include_coords = False)
    
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list)
    node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)
    np.savez_compressed(data_path+"pcno_quad_data.npz", \
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
                        node_measures_raw = node_measures_raw, \
                        node_measures=node_measures, node_weights=node_weights, \
                        node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights, \
                        features=features, \
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 
    
    # save nodes_list, elems_list, features_list
    np.savez_compressed(data_path+"pcno_quad_data_list.npz", \
                        nodes_list=nodes_list, elems_list=elems_list, features_list=features_list)


def load_data(data_path):
    equal_weights = True

    data = np.load(data_path+"pcno_quad_data.npz")
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
    node_measures = data["node_measures"]
    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
    features = data["features"]

    node_measures_raw = data["node_measures_raw"]
    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices]/node_measures[indices]
    return nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights


def data_preparition(nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights):
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

    n_train, n_test = 1000, 200


    x_train, x_test = torch.cat((nodes_input[:n_train, ...], node_rhos[:n_train, ...]), -1), torch.cat((nodes_input[-n_test:, ...], node_rhos[-n_test:, ...]),-1)
    aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
    aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])
    y_train, y_test = features[:n_train,...],    features[-n_test:,...]


    return x_train, x_test, aux_train, aux_test, y_train, y_test

def data_preparition_with_tensordict(nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights):
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

    n_train, n_test = 1000, 200


    train_data = TensorDict(
        {   "y": features[:n_train,...],
            "condition": TensorDict(
                {
                    "x": torch.cat((nodes_input[:n_train, ...], node_rhos[:n_train, ...]), -1),
                    "node_mask": node_mask[0:n_train,...],
                    "nodes": nodes[0:n_train,...],
                    "node_weights": node_weights[0:n_train,...],
                    "directed_edges": directed_edges[0:n_train,...],
                    "edge_gradient_weights": edge_gradient_weights[0:n_train,...]
                },
                batch_size=(n_train,),
            ),
        },
        batch_size=(n_train,),
    )
    test_data = TensorDict(
        {   "y": features[-n_test:,...],
            "condition": TensorDict(
                {
                    "x": torch.cat((nodes_input[-n_test:, ...], node_rhos[-n_test:, ...]),-1),
                    "node_mask": node_mask[-n_test:,...],
                    "nodes": nodes[-n_test:,...],
                    "node_weights": node_weights[-n_test:,...],
                    "directed_edges": directed_edges[-n_test:,...],
                    "edge_gradient_weights": edge_gradient_weights[-n_test:,...]
                },
                batch_size=(n_test,),
            ),
        },
        batch_size=(n_test,),
    )

    n_train, n_test = train_data["condition"]["x"].shape[0], test_data["condition"]["x"].shape[0]

    if config.parameter.normalization_x:
        x_normalizer = UnitGaussianNormalizer(train_data["x"], non_normalized_dim = config.parameter.non_normalized_dim_x, normalization_dim=config.parameter.normalization_dim_x)
        x_train = x_normalizer.encode(train_data["condition"]["x"])
        x_test = x_normalizer.encode(test_data["condition"]["x"])
        x_normalizer.to(device)
    else:
        x_normalizer = None
        
    if config.parameter.normalization_y:
        y_normalizer = UnitGaussianNormalizer(train_data["y"], non_normalized_dim = config.parameter.non_normalized_dim_y, normalization_dim=config.parameter.normalization_dim_y)
        y_train = y_normalizer.encode(train_data["y"])
        y_test = y_normalizer.encode(test_data["y"])
        y_normalizer.to(device)
    else:
        y_normalizer = None

    train_dataset = TensorDictDataset(keys=["y", "condition"], max_size=n_train)
    test_dataset = TensorDictDataset(keys=["y", "condition"], max_size=n_test)
    train_dataset.append(train_data, batch_size=n_train)
    test_dataset.append(test_data, batch_size=n_test)

    return train_dataset, test_dataset, x_normalizer, y_normalizer



def model_initialization(device, x_train, y_train):

    kx_max, ky_max = 32, 16
    ndims = 2
    Lx = Ly = 4.0

    modes = compute_Fourier_modes(ndims, [kx_max,ky_max], [Lx, Ly])
    modes = torch.tensor(modes, dtype=torch.float).to(device)

    flow_model_config = EasyDict(
        dict(
            device=device,
            gaussian_process=dict(
                type="Matern",
                args=dict(
                    length_scale=0.1, #0.01,
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
                            layers=[128,128,128,128,128],
                            fc_dim=128,
                            in_dim=y_train.shape[-1]+1+x_train.shape[-1], 
                            out_dim=y_train.shape[-1],
                            train_sp_L="together",
                            act='gelu'
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

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with=None, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    state = AcceleratorState()

    # Get the process rank
    process_rank = state.process_index
    set_seed(seed=42+process_rank)
    print(f"Process rank: {process_rank}")

    project_name = "PCNO_airfoil"

    config = EasyDict(
        dict(
            device=device,
            parameter=dict(
                batch_size=4,
                warmup_steps=1000 // 4 * 2000 // 100,
                learning_rate=5e-5 * accelerator.num_processes,
                iterations=1000 // 4 * 2000,
                log_rate=100,
                eval_rate=1000 // 4 * 500,
                checkpoint_rate=1000 // 4 * 500,
                video_save_path=f"output/{project_name}/videos",
                model_save_path=f"output/{project_name}/models",
                model_load_path=f"output/{project_name}/models/model_50000.pth",
                normalization_x=False,
                normalization_y=False,
                normalization_dim_x=[],
                normalization_dim_y=[],
                non_normalized_dim_x=3,
                non_normalized_dim_y=0,
            ),
        )
    )

    data_path = "/mnt/d/Dataset/pcno/airfoil/"
    # data_preprocess(data_path)
    nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights = load_data(data_path)

    train_dataset, test_dataset, x_normalizer, y_normalizer = data_preparition_with_tensordict(nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights)


    flow_model, flow_model_config = model_initialization(device, train_dataset["condition"]["x"], train_dataset["y"])

    if config.parameter.model_load_path is not None and os.path.exists(
        config.parameter.model_load_path
    ):
        # pop out _metadata key
        state_dict = torch.load(config.parameter.model_load_path, map_location="cpu")
        state_dict.pop("_metadata", None)
        # Create a new dictionary with updated keys
        prefix = "_orig_mod."
        new_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith(prefix):
                # Remove the prefix from the key
                new_key = key[len(prefix) :]
            else:
                new_key = key
            new_state_dict[new_key] = value

        flow_model.model.load_state_dict(new_state_dict)
        print("Model loaded from: ", config.parameter.model_load_path)



    flow_model.model = accelerator.prepare(flow_model.model)

    train_replay_buffer = TensorDictReplayBuffer(
        storage=train_dataset.storage,
        batch_size=config.parameter.batch_size,
        sampler=RandomSampler(),
        prefetch=10,
    )

    test_replay_buffer = TensorDictReplayBuffer(
        storage=test_dataset.storage,
        batch_size=config.parameter.batch_size,
        sampler=RandomSampler(),
        prefetch=10,
    )

    for iteration in track(
        range(config.parameter.iterations),
        description="Evaluating",
        disable=not accelerator.is_local_main_process,
    ):

        data = train_replay_buffer.sample()
        data = data.to(device)

        matern_kernel = matern_halfinteger_kernel_batchwise(
            X1=data["condition"]["nodes"],
            X2=data["condition"]["nodes"],
            length_scale=flow_model_config.gaussian_process.args.length_scale,
            nu=flow_model_config.gaussian_process.args.nu,
            variance=1.0,
        )

        def sample_from_covariance(C, D):
            # Compute Cholesky decomposition; shape [B, N, N]
            # L = torch.linalg.cholesky(C+1e-6*torch.eye(C.size(1), device=C.device, dtype=C.dtype).unsqueeze(0)) # for length scale 0.01
            L = torch.linalg.cholesky(C+1e-1*torch.eye(C.size(1), device=C.device, dtype=C.dtype).unsqueeze(0)) # for length scale 0.1
            # L = torch.linalg.cholesky(C+1e-2*torch.eye(C.size(1), device=C.device, dtype=C.dtype).unsqueeze(0)) # for length scale 0.5
            
            # Generate standard normal noise; shape [B, N, D]
            z = torch.randn(C.size(0), C.size(1), D*2, device=C.device, dtype=C.dtype)
            
            # Batched matrix multiplication; result shape [B, N, 2D]
            samples = L @ z

            # split the samples into two parts
            samples = torch.split(samples, [D, D], dim=-1)
            
            return samples[0], samples[1]

        # gaussian_process = flow_model.gaussian_process(data["nodes"])
        x0, gaussian_process_samples = sample_from_covariance(matern_kernel, data["y"].shape[-1])

        if y_normalizer is not None:
            x1 = y_normalizer.encode(data["y"])
        else:
            x1 = data["y"]

        loss = flow_model.functional_flow_matching_loss(x0=x0, x1=x1, condition=data["condition"], gaussian_process_samples=gaussian_process_samples)

        sampled_process = flow_model.sample_process(
            x0=x0,
            t_span=torch.linspace(0.0, 1.0, 100),
            condition=data["condition"],
            # with_grad=True,
        )

        x1_sampled = sampled_process[-1]


        # plt.pcolormesh(data["condition"]["nodes"][0,:,0].cpu().numpy().reshape(221,51), data["condition"]["nodes"][0,:,1].cpu().numpy().reshape(221,51), x0[0].cpu().numpy().reshape(221,51), shading='nearest')
        # plt.pcolormesh(data["condition"]["nodes"][0,:,0].cpu().numpy().reshape(221,51), data["condition"]["nodes"][0,:,1].cpu().numpy().reshape(221,51), x1_sampled[0].cpu().numpy().reshape(221,51), shading='gouraud')
        # plt.tripcolor(data["condition"]["nodes"][0,:,0].cpu().numpy(), data["condition"]["nodes"][0,:,1].cpu().numpy(), x1_sampled[0, :, -1].cpu().numpy(),  triangles=elems[sample_idx], shading='flat', norm=norm) 
        
        # plot 2 subplots for x1_sampled and x1, control x range from -0.5 to 1.5, y range from -0.5 to 0.5
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Iteration: {iteration}")
        axs[0].set_title("x1_sampled")
        axs[0].set_xlim(-0.5, 1.5)
        axs[0].set_ylim(-0.5, 0.5)
        axs[0].set_aspect('equal')
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].set_xticks(np.arange(-0.5, 1.5, 0.25))
        axs[0].set_yticks(np.arange(-0.5, 0.5, 0.25))
        axs[0].pcolormesh(data["condition"]["nodes"][0,:,0].cpu().numpy().reshape(221,51), data["condition"]["nodes"][0,:,1].cpu().numpy().reshape(221,51), x1_sampled[0].cpu().numpy().reshape(221,51), shading='gouraud')
        axs[1].set_title("x1")
        axs[1].set_xlim(-0.5, 1.5)
        axs[1].set_ylim(-0.5, 0.5)
        axs[1].set_aspect('equal')
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        axs[1].set_xticks(np.arange(-0.5, 1.5, 0.25))
        axs[1].set_yticks(np.arange(-0.5, 0.5, 0.25))
        axs[1].pcolormesh(data["condition"]["nodes"][0,:,0].cpu().numpy().reshape(221,51), data["condition"]["nodes"][0,:,1].cpu().numpy().reshape(221,51), x1[0].cpu().numpy().reshape(221,51), shading='gouraud')

        # color bar show value range for these two plots
        fig.colorbar(axs[0].collections[0], ax=axs[0], label="x1_sampled")
        fig.colorbar(axs[1].collections[0], ax=axs[1], label="x1")
        fig.tight_layout()

        # save fig as png
        # plt.savefig(f"output/{project_name}/iteration_{iteration}.png")

        plt.show()
        b = 1


