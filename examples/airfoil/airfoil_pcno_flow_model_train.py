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


def data_preprocess(data_path, file_name="pcno_quad_data.npz"):
    coordx = np.load(os.path.join(data_path, "NACA_Cylinder_X.npy"))
    coordy = np.load(os.path.join(data_path, "NACA_Cylinder_Y.npy"))
    data_out = np.load(os.path.join(data_path, "NACA_Cylinder_Q.npy"))[
        :, 4, :, :
    ]  # density, velocity 2d, pressure, mach number

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
    set_seed(seed=seed + process_rank)
    print(f"Process rank: {process_rank}")

    # check GPU brand, if NVIDIA RTX 4090 use batch size 4, if NVIDIA A100 use batch size 16
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "A100" in gpu_name or "A800" in gpu_name:
            batch_size = 16
        elif "4090" in gpu_name:
            batch_size = 4
        else:
            batch_size = 4
        print(f"GPU name: {gpu_name}, batch size: {batch_size}")
    else:
        gpu_name = "CPU"
        batch_size = 4
        print(f"CPU, batch size: {batch_size}")

    if args.data_preprocess:
        data_preprocess(args.data_path, file_name="pcno_quad_data.npz")
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
                learning_rate=5e-5 * accelerator.num_processes,
                iterations=iterations,
                log_rate=100,
                eval_rate=eval_rate,
                checkpoint_rate=checkpoint_rate,
                video_save_path=f"output/{project_name}/videos",
                model_save_path=f"output/{project_name}/models",
                model_load_path=None,
            ),
        )
    )

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
                        + 1e-6
                        * torch.eye(
                            C.size(1), device=C.device, dtype=C.dtype
                        ).unsqueeze(0)
                    )

                    # Generate standard normal noise; shape [B, N, D]
                    z = torch.randn(
                        C.size(0), C.size(1), D * 2, device=C.device, dtype=C.dtype
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
                x1_sampled = flow_model.sample(
                    x0=x0,
                    t_span=torch.linspace(0.0, 1.0, 100),
                    condition=data["condition"],
                )

                def plot_2d(data, x1_sampled, x1, title):
                    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                    fig.suptitle(f"Iteration: {iteration}, {title}")
                    axs[0].set_title("x1_sampled")
                    axs[0].set_xlim(-0.5, 1.5)
                    axs[0].set_ylim(-0.5, 0.5)
                    axs[0].set_aspect("equal")
                    axs[0].set_xlabel("x")
                    axs[0].set_ylabel("y")
                    axs[0].set_xticks(np.arange(-0.5, 1.5, 0.25))
                    axs[0].set_yticks(np.arange(-0.5, 0.5, 0.25))
                    axs[0].pcolormesh(
                        data["condition"]["nodes"][0, :, 0]
                        .cpu()
                        .numpy()
                        .reshape(221, 51),
                        data["condition"]["nodes"][0, :, 1]
                        .cpu()
                        .numpy()
                        .reshape(221, 51),
                        x1_sampled[0].cpu().numpy().reshape(221, 51),
                        shading="gouraud",
                    )
                    axs[1].set_title("x1")
                    axs[1].set_xlim(-0.5, 1.5)
                    axs[1].set_ylim(-0.5, 0.5)
                    axs[1].set_aspect("equal")
                    axs[1].set_xlabel("x")
                    axs[1].set_ylabel("y")
                    axs[1].set_xticks(np.arange(-0.5, 1.5, 0.25))
                    axs[1].set_yticks(np.arange(-0.5, 0.5, 0.25))
                    axs[1].pcolormesh(
                        data["condition"]["nodes"][0, :, 0]
                        .cpu()
                        .numpy()
                        .reshape(221, 51),
                        data["condition"]["nodes"][0, :, 1]
                        .cpu()
                        .numpy()
                        .reshape(221, 51),
                        x1[0].cpu().numpy().reshape(221, 51),
                        shading="gouraud",
                    )

                    # color bar show value range for these two plots
                    fig.colorbar(axs[0].collections[0], ax=axs[0], label="x1_sampled")
                    fig.colorbar(axs[1].collections[0], ax=axs[1], label="x1")
                    fig.tight_layout()

                    # save fig as png
                    plt.savefig(f"output/{project_name}/{title}_iteration_{iteration}.png")
                    fig.clear()
                    plt.close(fig)

                plot_2d(data, x1_sampled, x1, "train_data")

                data_test = test_replay_buffer.sample()
                data_test = data_test.to(device)

                if y_normalizer is not None:
                    x1_test = y_normalizer.encode(data_test["y"])
                else:
                    x1_test = data_test["y"]

                x1_sampled_test = flow_model.sample(
                    x0=x0,
                    t_span=torch.linspace(0.0, 1.0, 100),
                    condition=data_test["condition"],
                )
                plot_2d(data_test, x1_sampled_test, x1_test, "test_data")

                to_log["reconstruction_error_train_dataset"] = torch.mean(
                    torch.abs(x1_sampled - x1)
                ).item()
                to_log["reconstruction_error_test_dataset"] = torch.mean(
                    torch.abs(x1_sampled_test - x1_test)
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
