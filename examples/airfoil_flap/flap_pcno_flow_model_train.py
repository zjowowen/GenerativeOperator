import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer
from torchrl.data import RandomSampler

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed
from easydict import EasyDict
from rich.progress import track
from easydict import EasyDict

from generative_operator.model.point_cloud_flow_model import PointCloudFunctionalFlow
from generative_operator.utils.optimizer import CosineAnnealingWarmupLR
from generative_operator.dataset.tensordict_dataset import TensorDictDataset

from generative_operator.neural_networks.neural_operators.point_cloud_neural_operator import (
    preprocess_data,
    compute_node_weights,
    compute_Fourier_modes,
)
from generative_operator.neural_networks.neural_operators.point_cloud_data_process import (
    convert_structured_data,
)

from generative_operator.gaussian_process.matern import (
    matern_halfinteger_kernel_batchwise,
)
from generative_operator.utils.normalizer import UnitGaussianNormalizer


# def data_preprocess(data_path):
#     coordx    = np.load(data_path+"NACA_Cylinder_X.npy")
#     coordy    = np.load(data_path+"NACA_Cylinder_Y.npy")
#     data_out  = np.load(data_path+"NACA_Cylinder_Q.npy")[:,4,:,:] #density, velocity 2d, pressure, mach number

#     nodes_list, elems_list, features_list = convert_structured_data([coordx, coordy], data_out[...,np.newaxis], nnodes_per_elem = 4, feature_include_coords = False)
    
#     nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list)
#     node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
#     node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)
#     np.savez_compressed(data_path+"pcno_quad_data.npz", \
#                         nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
#                         node_measures_raw = node_measures_raw, \
#                         node_measures=node_measures, node_weights=node_weights, \
#                         node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights, \
#                         features=features, \
#                         directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 


def load_data(data_path, file_name="pcno_triangle_data.npz"):
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
    node_rhos[indices] = node_rhos[indices]/node_measures[indices]
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
    n_train=500,
    n_test=40,
    normalization_x=False,
    normalization_y=False,
    normalization_dim_x=[],
    normalization_dim_y=[],
    non_normalized_dim_x=1,   # to check
    non_normalized_dim_y=0,
):
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

    nodes_input = torch.cat([nodes_input, node_rhos], dim=-1)

    ndata = nodes_input.shape[0]
    ndata1 = 1931
    ndata2 = 1932
    m_train, m_test = n_train // 2, n_test // 2

    train_type = "flap"
    train_index = torch.arange(n_train)

    test_index = torch.cat(
        [torch.arange(ndata1 - m_test, ndata1), torch.arange(ndata1, ndata1 + m_test)], dim=0
    )

    feature_type = "mach"
    feature_type_index = 1

    train_data = TensorDict(
        {   "y": features[train_index,:,1:2],
            "condition": TensorDict(
                {
                    "x": nodes_input[train_index, ...],
                    "nnodes": nnodes[train_index, ...],
                    "node_mask": node_mask[train_index, ...],
                    "nodes": nodes[train_index, ...],
                    "node_weights": node_weights[train_index, ...],
                    "directed_edges": directed_edges[train_index, ...],
                    "edge_gradient_weights": edge_gradient_weights[train_index, ...]
                },
                batch_size=(n_train,),
            ),
        },
        batch_size=(n_train,),
    )
    test_data = TensorDict(
        {   "y": features[test_index,:,1:2],
            "condition": TensorDict(
                {
                    "x": nodes_input[test_index, ...],
                    "nnodes": nnodes[test_index, ...],
                    "node_mask": node_mask[test_index, ...],
                    "nodes": nodes[test_index, ...],
                    "node_weights": node_weights[test_index, ...],
                    "directed_edges": directed_edges[test_index, ...],
                    "edge_gradient_weights": edge_gradient_weights[test_index, ...]
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
            non_normalized_dim = non_normalized_dim_x, 
            normalization_dim= normalization_dim_x,
        )
        x_normalizer.to(device)
    else:
        x_normalizer = None
        
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(
            train_data["y"], 
            non_normalized_dim = non_normalized_dim_y, 
            normalization_dim = normalization_dim_y,
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
    device, 
    x_train, 
    y_train,
    kx_max=16,
    ky_max=16,
    Lx=1,
    Ly=0.5,
    ndims=2,
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
                            layers=[128,128,128,128,128],
                            fc_dim=128,
                            in_dim=y_train.shape[-1] + 1 + x_train.shape[-1], 
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

    import argparse

    parser = argparse.ArgumentParser(description="PCNO flow model training")
    parser.add_argument(
        "--project_name",
        type=str,
        default="PCNO_airfoil_flap",
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
        default="../../../data/airfoil_flap/",  # "data/",
        help="Path to the data",
    )
    parser.add_argument(
        "--data_preprocess",
        action="store_true",
        help="Whether to preprocess the data",
    )
    parser.add_argument(
        "--wandb",
        action="store_false",
        help="Whether to use wandb",
    )
    args = parser.parse_args()
    project_name = args.project_name
    seed = args.seed

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="wandb" if args.wandb else None, kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device
    state = AcceleratorState()

    # Get the process rank
    process_rank = state.process_index
    set_seed(seed=42+process_rank)
    print(f"Process rank: {process_rank}")

    # check GPU brand, if NVIDIA RTX 4090 use batch size 4, if NVIDIA A100 use batch size 16
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "A100" in gpu_name or "A800" in gpu_name:
            batch_size = 2
        elif "4090" in gpu_name:
            batch_size = 1
        else:
            batch_size = 1
        print(f"GPU name: {gpu_name}, batch size: {batch_size}")
    else:
        gpu_name = "CPU"
        batch_size = 1
        print(f"CPU, batch size: {batch_size}")

    if args.data_preprocess:
        data_preprocess(args.data_path, file_name="pcno_triangle_data.npz")

    data_path = args.data_path

    (
        nnodes,
        node_mask,
        nodes,
        node_weights,
        node_rhos,
        features,
        directed_edges,
        edge_gradient_weights,
    ) = load_data(data_path)

    (
        train_dataset,
        test_dataset,
        x_normalizer,
        y_normalizer,
    ) = data_preparition_with_tensordict(
        nnodes,
        node_mask,
        nodes,
        node_weights,
        node_rhos,
        features,
        directed_edges,
        edge_gradient_weights,
    )

    flow_model, flow_model_config = model_initialization(
        device, train_dataset["condition"]["x"], train_dataset["y"]
    )

    # Set the number of iterations and warmup steps
    data_size = len(train_dataset)
    iterations = data_size // batch_size * 2000 + 1
    warmup_steps = iterations // 100
    eval_rate = iterations // 100
    checkpoint_rate = iterations // 100
    iterations_per_epoch = data_size // batch_size // accelerator.num_processes

    config = EasyDict(
        dict(
            device=device,
            parameter=dict(
                batch_size=batch_size,
                warmup_steps=warmup_steps,
                learning_rate=2.5e-5 * accelerator.num_processes,
                iterations=iterations,
                log_rate=100,
                eval_rate=eval_rate,
                checkpoint_rate=checkpoint_rate,
                video_save_path=f"output/{project_name}/videos",
                model_save_path=f"output/{project_name}/2.5lr/models",
                model_load_path=None,
            ),
        )
    )


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

    # Visualization

    path_eval = os.path.join(data_path, "Airfoil_flap_data/fluid_mesh")

    elems_eval = np.load(path_eval + "/elems_%05d" % (10) + ".npy")
    nodes_eval = np.load(path_eval + "/nodes_%05d" % (10) + ".npy")
    features_eval = np.load(path_eval + "/features_%05d" % (10) + ".npy")
    
    path_test = os.path.join(data_path, "Airfoil_flap_data/fluid_mesh")
    elems_test = np.load(path_test + "/elems_%05d" % (1911) + ".npy")
    nodes_test = np.load(path_test + "/nodes_%05d" % (1911) + ".npy")
    features_test = np.load(path_test + "/features_%05d" % (1911) + ".npy")

    def plot_2d(data, x_sampled, x_truth,nodes,elems,features, title):
        '''
        x_sampeled: torch.Size([1,nnodes,1]):  sampled features from flow_model.sample() 
        x_truth: torch.Size([1,nnodes,1]): ground truth features from dateset data["y"] 
        nodes (nnodes, 2) :x,y coordinates
        elems (xxxx,3) :the triangles formed by the nodes
        features (nnodes,3): Directly extract from the corresponding .npy file.
        0:pressure  
        1:mach number   
        2:indicator:  3 values
              0: innner nodes
              1: boundary codes
              2: farfield codes

        '''       
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        # fig.suptitle(f"Iteration: {iteration}, {title}")
        x1, x2, y1, y2 = -0.5, 1.5, -0.5, 1.5 


        # z :(nnodes,1) mach number
        z_sampled = x_sampled.detach().cpu().numpy()
        z_truth = x_truth.detach().cpu().numpy()
        
        
        # 计算共享的颜色范围
        vmin = min(z_sampled.min(), z_truth.min())
        vmax = max(z_sampled.max(), z_truth.max())

        # 在左侧子图中绘制第一个流体场
        ax1.set_aspect('equal')
        tpc1 = ax1.tripcolor(nodes[:,0], nodes[:,1], z_sampled[:,0], triangles=elems, shading='flat',vmin = vmin, vmax = vmax)
        ax1.scatter(nodes[features[:,2]==1, 0], nodes[features[:,2]==1, 1], color="red", linewidths=0.01)
        ax1.scatter(nodes[features[:,2]==2, 0], nodes[features[:,2]==2, 1], color="black", linewidths=0.01)
        fig.colorbar(tpc1, ax=ax1)
        ax1.set_title("x1_sample")
        
        # 添加放大视图
        axins1 = ax1.inset_axes(
            [0.3, 0.6, 0.6, 0.6],
            xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
        axins1.tripcolor(nodes[:,0], nodes[:,1], z_sampled[:,0] , triangles=elems, shading='flat',vmin = vmin, vmax = vmax)
        ax1.indicate_inset_zoom(axins1, edgecolor="black")
        
        # 在右侧子图中绘制第二个流体场
        ax2.set_aspect('equal')
        tpc2 = ax2.tripcolor(nodes[:,0], nodes[:,1], features[:,1], triangles=elems, shading='flat',vmin = vmin, vmax = vmax)
        ax2.scatter(nodes[features[:,2]==1, 0], nodes[features[:,2]==1, 1], color="red", linewidths=0.01)
        ax2.scatter(nodes[features[:,2]==2, 0], nodes[features[:,2]==2, 1], color="black", linewidths=0.01)
        fig.colorbar(tpc2, ax=ax2)
        ax2.set_title("x1")
        
        # 添加放大视图
        axins2 = ax2.inset_axes(
            [0.3, 0.6, 0.6, 0.6],
            xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
        axins2.tripcolor(nodes[:,0], nodes[:,1], features[:,1], triangles=elems, shading='flat',vmin = vmin, vmax = vmax)
        ax2.indicate_inset_zoom(axins2, edgecolor="black")
        
        plt.tight_layout()

        # save fig as png
        plt.savefig(f"output/{project_name}/2.5lr/{title}_iteration_{iteration}.png")
        fig.clear()
        plt.close(fig)

    accelerator.init_trackers(project_name=project_name, config=None)
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
                    L = torch.linalg.cholesky(
                        C
                        + 1e-2 * torch.eye(
                                    C.size(1), 
                                    device=C.device, 
                                    dtype=C.dtype
                                ).unsqueeze(0)
                    )
                    
                    # Generate standard normal noise; shape [B, N, D]
                    z = torch.randn(
                            C.size(0), 
                            C.size(1), 
                            D*2, 
                            device=C.device, 
                            dtype=C.dtype
                    )
                    
                    # Batched matrix multiplication; result shape [B, N, 2D]
                    samples = L @ z

                    # split the samples into two parts
                    samples = torch.split(samples, [D, D], dim=-1)
                    
                    return samples[0], samples[1]

                # gaussian_process = flow_model.gaussian_process(data["nodes"])
                x0, gaussian_process_samples = sample_from_covariance(
                    matern_kernel, data["y"].shape[-1]
                )

                if y_normalizer is not None:
                    x1 = y_normalizer.encode(data["y"])
                else:
                    x1 = data["y"]

                loss = flow_model.functional_flow_matching_loss(
                    x0=x0, 
                    x1=x1, 
                    condition=data["condition"], 
                    gaussian_process_samples=gaussian_process_samples, 
                    mse_loss=True,
                )
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

        loss = accelerator.gather(loss)
        if accelerator.is_local_main_process:
            to_log = {}
            if iteration % config.parameter.log_rate == 0:
                to_log["iteration"] = iteration
                to_log["epoch"] = iteration // iterations_per_epoch
                to_log["lr"] = scheduler.get_last_lr()[0]

                if len(loss.shape) == 0:
                    to_log["loss/std"] = 0
                    to_log["loss/0"] = loss.item()
                elif loss.shape[0] > 1:
                    to_log["loss/std"] = loss.std().item()
                    for i in range(loss.shape[0]):
                        to_log[f"loss/{i}"] = loss[i].item()
                acc_train_loss = loss.mean().item()
                print(
                    f"iteration: {iteration}, epoch: {iteration // iterations_per_epoch}, train_loss: {acc_train_loss:.5f}, lr: {scheduler.get_last_lr()[0]:.7f}"
                )

            if iteration % config.parameter.eval_rate == 0:
                flow_model.eval()
                data_eval = train_dataset[10:11]
                data_eval = data_eval.to(device)

                if y_normalizer is not None:
                    x_eval = y_normalizer.encode(data_eval["y"])
                else:
                    x_eval = data_eval["y"]

                x1_eval_sampled = flow_model.sample(
                    x0=x0[0:1],
                    t_span=torch.linspace(0.0, 1.0, 100),
                    condition=data_eval["condition"],
                )
                # print("x1_eval_sampled.shape", x1_eval_sampled.shape)
                # print("x_eval.shape", x_eval.shape)
                n_eval = data_eval["condition"]["nnodes"]
                x_eval_nnodes = x1_eval_sampled[:n_eval[0],:]

                plot_2d(data_eval, x_eval_nnodes, x_eval_nnodes,nodes_eval,elems_eval,features_eval, "train_data")

                data_test = test_dataset[0:1]
                data_test = data_test.to(device)

                if y_normalizer is not None:
                    x_test = y_normalizer.encode(data_test["y"])
                else:
                    x_test = data_test["y"]

                x1_test_sampled = flow_model.sample(
                    x0=x0[0:1],
                    t_span=torch.linspace(0.0, 1.0, 100),
                    condition=data_test["condition"],
                )

                n_test = data_test["condition"]["nnodes"]
                x_test_nnodes = x1_test_sampled[:n_test[0],:]

                plot_2d(data_test, x_test_nnodes, x_test_nnodes,nodes_test,elems_test,features_test, "test_data")

                

                x1_test_sampled = flow_model.sample(
                    x0=x0[0:1],
                    t_span=torch.linspace(0.0, 1.0, 100),
                    condition=data_test["condition"],
                )
                
                to_log["reconstruction_error_train_dataset"] = torch.mean(
                    torch.abs(x1_eval_sampled - x_eval)
                ).item()
                to_log["reconstruction_error_test_dataset"] = torch.mean(
                    torch.abs(x1_test_sampled - x_test)
                ).item()

            if len(list(to_log.keys())) > 0:
                accelerator.log(
                    to_log,
                    step=iteration,
                )

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


        