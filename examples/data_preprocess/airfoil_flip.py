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
from generative_operator.utils.optimizer import CosineAnnealingWarmupLR
from generative_operator.dataset.tensordict_dataset import TensorDictDataset

from generative_operator.neural_networks.neural_operators.point_cloud_neural_operator import preprocess_data, compute_node_weights, compute_Fourier_modes
from generative_operator.neural_networks.neural_operators.point_cloud_data_process import compute_triangle_area_, compute_tetrahedron_volume_, compute_measure_per_elem_, compute_node_measures, convert_structured_data

from generative_operator.gaussian_process.matern import matern_halfinteger_kernel_batchwise
from generative_operator.utils.normalizer import UnitGaussianNormalizer


'''
Download airfoil with flap data from 
- PKU drive: https://disk.pku.edu.cn/link/AA53109CB764334A63A6ADF3E09EABBAE7
- Name of the data file: airfoil_flap.zip
'''

data_path = "/mnt/d/Dataset/"

def load_data(data_path):

    path1 = os.path.join(data_path, "Airfoil_flap_data/fluid_mesh")
    path2 = os.path.join(data_path, "Airfoil_data/fluid_mesh")

    ndata1, ndata2 = 1931, 1932
    elem_dim = 2

    nodes_list = []
    elems_list = []
    features_list = []

    for i in range(ndata1):
        nodes_list.append(np.load(path1 + "/nodes_%05d" % (i) + ".npy"))
        elems = np.load(path1 + "/elems_%05d" % (i) + ".npy")
        elems_list.append(np.concatenate((np.full((elems.shape[0], 1), elem_dim, dtype=int), elems), axis=1))
        features_list.append(np.load(path1 + "/features_%05d" % (i) + ".npy"))

    for i in range(ndata2):
        nodes_list.append(np.load(path2 + "/nodes_%05d" % (i) + ".npy"))

        elems = np.load(path2 + "/elems_%05d" % (i) + ".npy")
        elems_list.append(np.concatenate((np.full((elems.shape[0], 1), elem_dim, dtype=int), elems), axis=1))
        features_list.append(np.load(path2 + "/features_%05d" % (i) + ".npy"))

    return nodes_list, elems_list, features_list

print("Loading data")
nodes_list, elems_list, feats_list = load_data(data_path)
nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data(
    nodes_list, elems_list, feats_list)
print(f"nodes{nodes.shape}", flush=True)
node_measures, node_weights = compute_node_weights(nnodes, node_measures_raw, equal_measure = False)
node_equal_measures, node_equal_weights = compute_node_weights(nnodes, node_measures_raw, equal_measure = True)
np.savez_compressed(data_path+"pcno_triangle_data.npz",
                    nnodes=nnodes, node_mask=node_mask, nodes=nodes,
                    node_measures_raw = node_measures_raw,
                    node_measures=node_measures, node_weights=node_weights,
                    node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights,
                    features=features,
                    directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 