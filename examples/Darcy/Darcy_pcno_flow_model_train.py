import datetime
import argparse

import os
import torch.multiprocessing as mp


import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import math

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

from scipy.io import loadmat



def data_preprocess(data_path):

    data1 = loadmat(data_path+"piececonst_r421_N1024_smooth1")
    data2 = loadmat(data_path+"piececonst_r421_N1024_smooth2")

    indices = np.concatenate((np.arange(0, 1000), np.arange(2048 - 200, 2048)))
    data_in = np.vstack((data1["coeff"], data2["coeff"]))[indices, 0::downsample_ratio, 0::downsample_ratio]  # shape: 1200,421,421
    data_out = np.vstack((data1["sol"], data2["sol"]))[indices, 0::downsample_ratio, 0::downsample_ratio]     # shape: 1200,421,421

    # data_in = np.vstack((data1["coeff"], data2["coeff"]))[:, 0::downsample_ratio, 0::downsample_ratio]  # shape: 2048,421,421
    # data_out = np.vstack((data1["sol"], data2["sol"]))[:, 0::downsample_ratio, 0::downsample_ratio]     # shape: 2048,421,421
    features = np.stack((data_in, data_out), axis=3)
    ndata = data_in.shape[0]

    Np = data_in.shape[1]
    L = 1.0
    grid_1d = np.linspace(0, L, Np)
    grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)
    grid_x, grid_y = grid_x.T, grid_y.T

    nodes_list, elems_list, features_list = convert_structured_data([np.tile(grid_x, (ndata, 1, 1)), np.tile(grid_y, (ndata, 1, 1))], features, nnodes_per_elem = 4, feature_include_coords = False)
    #uniform weights
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list)
    node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)
    np.savez_compressed(data_path+"pcno_quad_data_"+str(downsample_ratio)+".npz", \
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
                        node_measures_raw = node_measures_raw, \
                        node_measures=node_measures, node_weights=node_weights, \
                        node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights, \
                        features=features, \
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 


def load_data(data_path):
    equal_weights = False

    data = np.load(data_path+"pcno_quad_data_"+str(downsample_ratio)+".npz")
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
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

    # This is important
    nodes_input = nodes.clone()

    n_train, n_test = 100, 40


    x_train, x_test = torch.cat((features[:n_train, :, [0]], nodes_input[:n_train, ...], node_rhos[:n_train, ...]),-1), torch.cat((features[-n_test:, :, [0]], nodes_input[-n_test:, ...], node_rhos[-n_test:, ...]),-1)
    aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
    aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])
    y_train, y_test = features[:n_train, :, [1]],       features[-n_test:, :, [1]]

    return x_train, x_test, aux_train, aux_test, y_train, y_test

def data_preparition_with_tensordict(nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights):
    print("Casting to tensor")
    nnodes = torch.from_numpy(nnodes)
    node_mask = torch.from_numpy(node_mask)
    nodes = torch.from_numpy(nodes.astype(np.float32))
    node_weights = torch.from_numpy(node_weights.astype(np.float32))
    node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
    features = torch.from_numpy(features.astype(np.float32))
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

    # This is important
    nodes_input = nodes.clone()

    n_train, n_test = 50, 40


    train_data = TensorDict(
        {   "y": features[:n_train, :, [1]],
            "condition": TensorDict(
                {
                    "x": torch.cat((features[:n_train, :, [0]], nodes_input[:n_train, ...], node_rhos[:n_train, ...]),-1),
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
        {   "y": features[-n_test:, :, [1]],
            "condition": TensorDict(
                {
                    "x": torch.cat((features[-n_test:, :, [0]], nodes_input[-n_test:, ...], node_rhos[-n_test:, ...]),-1),
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

    k_max = 16
    ndims = 2
    modes = compute_Fourier_modes(ndims, [k_max,k_max], [1.0,1.0])
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
                            layers=[128,128,128,128,128],
                            fc_dim=128,
                            in_dim=y_train.shape[-1]+1+x_train.shape[-1], 
                            out_dim=y_train.shape[-1],
                            train_sp_L="independently",
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

    project_name = "PCNO_Darcy_Square"

    # check GPU brand, if NVIDIA RTX 4090 use batch size 4, if NVIDIA A100 use batch size 16
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "A100" in gpu_name:
            batch_size = 16
        elif "4090" in gpu_name:
            batch_size = 4
        else:
            batch_size = 4
    else:
        batch_size = 4
    print(f"GPU name: {gpu_name}, batch size: {batch_size}")

    #checkpoint_rate=1000 // batch_size * 50,
    config = EasyDict(
        dict(
            device=device,
            parameter=dict(
                batch_size=batch_size,
                warmup_steps=5000,
                learning_rate=5e-5 * accelerator.num_processes,
                iterations=200001,
                log_rate=100,
                eval_rate=1000 // batch_size * 500,
                checkpoint_rate=100000,
                video_save_path=f"output/{project_name}/videos",
                model_save_path=f"output/{project_name}/models",
                model_load_path=None,
                normalization_x=False,
                normalization_y=False,
                normalization_dim_x=[],
                normalization_dim_y=[],
                non_normalized_dim_x=4,
                non_normalized_dim_y=0,
            ),
        )
    ) 

    data_path = "../../../NeuralOperator/data/darcy_square/"

    downsample_ratio = 2
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
        flow_model.model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(
        flow_model.model.parameters(), lr=config.parameter.learning_rate
    )

    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        T_max=config.parameter.iterations,
        eta_min=2e-6,
        warmup_steps=config.parameter.warmup_steps,
    )

    flow_model.model, optimizer = accelerator.prepare(flow_model.model, optimizer)

    os.makedirs(config.parameter.model_save_path, exist_ok=True)

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


    accelerator.init_trackers("PCNO_Darcy_flow", config=None)
    accelerator.print("✨ Start training ...")

    for iteration in track(
        range(config.parameter.iterations),
        description="Training",
        disable=not accelerator.is_local_main_process,
    ):
        flow_model.train()
        with accelerator.autocast():
            with accelerator.accumulate(flow_model.model):
                
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
                    L = torch.linalg.cholesky(C+1e-6*torch.eye(C.size(1), device=C.device, dtype=C.dtype).unsqueeze(0))
                    
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

                loss = flow_model.functional_flow_matching_loss(x0=x0, x1=x1, condition=data["condition"], gaussian_process_samples=gaussian_process_samples, mse_loss=True)
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()


        loss = accelerator.gather(loss)
        if iteration % config.parameter.log_rate == 0:
            if accelerator.is_local_main_process:
                to_log = {
                        "loss/mean": loss.mean().item(),
                        "iteration": iteration,
                        "lr": scheduler.get_last_lr()[0],
                    }
                
                if len(loss.shape) == 0:
                    to_log["loss/std"] = 0
                    to_log["loss/0"] = loss.item()
                elif loss.shape[0] > 1:
                    to_log["loss/std"] = loss.std().item()
                    for i in range(loss.shape[0]):
                        to_log[f"loss/{i}"] = loss[i].item()
                accelerator.log(
                    to_log,
                    step=iteration,
                )
                acc_train_loss = loss.mean().item()
                print(f"iteration: {iteration}, train_loss: {acc_train_loss:.5f}, lr: {scheduler.get_last_lr()[0]:.7f}")

        if iteration % config.parameter.eval_rate == 0:
            pass
            # sampled_process = flow_model.sample_process(
            #     x0=x0,
            #     t_span=torch.linspace(0.0, 1.0, 10),
            #     condition=data["condition"],
            #     with_grad=True,
            # )

        if iteration % config.parameter.checkpoint_rate == 0:
            if accelerator.is_local_main_process:
                if not os.path.exists(config.parameter.model_save_path):
                    os.makedirs(config.parameter.model_save_path)
                torch.save(
                    accelerator.unwrap_model(flow_model.model).state_dict(),
                    f"{config.parameter.model_save_path}/model_{iteration}.pth",
                )

        accelerator.wait_for_everyone()

    accelerator.print("✨ Training complete!")
    accelerator.end_training()