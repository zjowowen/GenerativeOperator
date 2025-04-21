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
Download flow over airfoil (Euler equation) data from 

Google drive:
https://drive.google.com/drive/folders/1JUkPbx0-lgjFHPURH_kp1uqjfRn3aw9-


NACA_Cylinder_X.npy
NACA_Cylinder_Y.npy
NACA_Cylinder_Q.npy
'''

data_path = "/mnt/d/Dataset/"


print("Loading Data")
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