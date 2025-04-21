'''
Download darcy equation data from 

Google drive
https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-


Darcy_421.zip
piececonst_r421_N1024_smooth1.mat
piececonst_r421_N1024_smooth2.mat
'''

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

from scipy.io import loadmat

data_path = "/mnt/d/Dataset/"

downsample_ratio = 2



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
