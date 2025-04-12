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

from generative_operator.dataset.tensordict_dataset import TensorDictDataset

from generative_operator.neural_networks.neural_operators.point_cloud_neural_operator import preprocess_data, compute_node_weights, compute_Fourier_modes
from generative_operator.neural_networks.neural_operators.point_cloud_data_process import compute_triangle_area_, compute_tetrahedron_volume_, compute_measure_per_elem_, compute_node_measures, convert_structured_data


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
        {
            "x": torch.cat((nodes_input[:n_train, ...], node_rhos[:n_train, ...]), -1),
            "y": features[:n_train,...],
            "node_mask": node_mask[0:n_train,...],
            "nodes": nodes[0:n_train,...],
            "node_weights": node_weights[0:n_train,...],
            "directed_edges": directed_edges[0:n_train,...],
            "edge_gradient_weights": edge_gradient_weights[0:n_train,...]
        },
        batch_size=(n_train,),
    )
    test_data = TensorDict(
        {
            "x": torch.cat((nodes_input[-n_test:, ...], node_rhos[-n_test:, ...]),-1),
            "y": features[-n_test:,...],
            "node_mask": node_mask[-n_test:,...],
            "nodes": nodes[-n_test:,...],
            "node_weights": node_weights[-n_test:,...],
            "directed_edges": directed_edges[-n_test:,...],
            "edge_gradient_weights": edge_gradient_weights[-n_test:,...]
        },
        batch_size=(n_test,),
    )

    return train_data, test_data



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
                            in_dim=x_train.shape[-1], 
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

    return model

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

    epochs = 500
    base_lr = 0.001
    lr_ratio = 10
    scheduler = "OneCycleLR"
    weight_decay = 1.0e-4
    batch_size=20

    normalization_x = False #True
    normalization_y = True
    normalization_dim_x = []
    normalization_dim_y = []
    non_normalized_dim_x = 3
    non_normalized_dim_y = 0

    config = EasyDict(
        dict(
            device=device,
            parameter=dict(
                train_samples=20000,
                batch_size=1024,
                learning_rate=5e-5 * accelerator.num_processes,
                iterations=20000 // 1024 * 2000,
                log_rate=100,
                eval_rate=20000 // 1024 * 500,
                checkpoint_rate=20000 // 1024 * 500,
                video_save_path=f"output/{project_name}/videos",
                model_save_path=f"output/{project_name}/models",
                model_load_path=None,
            ),
            train=dict(
                base_lr=base_lr,
                lr_ratio=lr_ratio,
                weight_decay=weight_decay,
                epochs=epochs,
                scheduler=scheduler,
                batch_size=batch_size,
                normalization_x=normalization_x,
                normalization_y=normalization_y,
                normalization_dim_x=normalization_dim_x,
                normalization_dim_y=normalization_dim_y,
                non_normalized_dim_x=non_normalized_dim_x,
                non_normalized_dim_y=non_normalized_dim_y
            ),
        )
    )

    data_path = "/mnt/d/Dataset/"
    # data_preprocess(data_path)
    nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights = load_data(data_path)
    # x_train, x_test, aux_train, aux_test, y_train, y_test = data_preparition(nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights)

    train_data, test_data = data_preparition_with_tensordict(nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights)

    flow_model = model_initialization(device, train_data["x"], train_data["y"])

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

    flow_model.model, optimizer = accelerator.prepare(flow_model.model, optimizer)

    os.makedirs(config.parameter.model_save_path, exist_ok=True)

    n_train, n_test = train_data["x"].shape[0], test_data["x"].shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y = config["train"]["normalization_x"], config["train"]["normalization_y"]
    normalization_dim_x, normalization_dim_y = config["train"]["normalization_dim_x"], config["train"]["normalization_dim_y"]
    non_normalized_dim_x, non_normalized_dim_y = config["train"]["non_normalized_dim_x"], config["train"]["non_normalized_dim_y"]
    

    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(train_data["x"], non_normalized_dim = non_normalized_dim_x, normalization_dim=normalization_dim_x)
        x_train = x_normalizer.encode(train_data["x"])
        x_test = x_normalizer.encode(test_data["x"])
        x_normalizer.to(device)
        
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(train_data["y"], non_normalized_dim = non_normalized_dim_y, normalization_dim=normalization_dim_y)
        y_train = y_normalizer.encode(train_data["y"])
        y_test = y_normalizer.encode(test_data["y"])
        y_normalizer.to(device)


    train_dataset = TensorDictDataset(keys=["x", "y", "node_mask", "nodes", "node_weights", "directed_edges", "edge_gradient_weights"], max_size=n_train)
    test_dataset = TensorDictDataset(keys=["x", "y", "node_mask", "nodes", "node_weights", "directed_edges", "edge_gradient_weights"], max_size=n_test)
    train_dataset.append(train_data, batch_size=n_train)
    test_dataset.append(test_data, batch_size=n_test)

    train_replay_buffer = TensorDictReplayBuffer(
        storage=train_dataset.storage,
        batch_size=config['train']['batch_size'],
        sampler=RandomSampler(),
        prefetch=10,
    )

    test_replay_buffer = TensorDictReplayBuffer(
        storage=test_dataset.storage,
        batch_size=config['train']['batch_size'],
        sampler=RandomSampler(),
        prefetch=10,
    )

    def get_data(dataloader):
        while True:
            yield from dataloader

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

                gaussian_process = flow_model.gaussian_process(data["nodes"])
                x0 = gaussian_process.sample()
                loss = flow_model.optimal_transport_functional_flow_matching_loss(x0=x0, x1=data["x"], condition=data["condition"])
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()


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

        if iteration % config.parameter.checkpoint_rate == 0:
            if accelerator.is_local_main_process:
                if not os.path.exists(config.parameter.model_save_path):
                    os.makedirs(config.parameter.model_save_path)
                torch.save(
                    accelerator.unwrap_model(flow_model.model).state_dict(),
                    f"{config.parameter.model_save_path}/model_{iteration}.pth",
                )

        accelerator.wait_for_everyone()

