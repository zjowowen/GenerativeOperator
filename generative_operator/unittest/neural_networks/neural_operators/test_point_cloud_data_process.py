import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from generative_operator.neural_networks.neural_operators.point_cloud_data_process import compute_triangle_area_, compute_tetrahedron_volume_, compute_measure_per_elem_
from generative_operator.neural_networks.neural_operators.point_cloud_data_process import compute_node_measures, compute_node_measures_torch
from generative_operator.neural_networks.neural_operators.point_cloud_data_process import convert_structured_data, convert_structured_data_torch



import unittest

####################
# pytorch version
####################

def test_compute_node_measures_torch():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Suppose we have 4 nodes in 2D
    # shape: (4, 2)
    nodes = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], device=device)
    
    # Suppose we have two elements:
    # 1) A line (elem_dim=1) connecting node indices 0,1
    # 2) A triangle (elem_dim=2) connecting node indices 0,1,2
    # We set -1 as padding after that when not needed
    elems = torch.tensor([
        [1, 0, 1, -1, -1],  # line: connect node 0 and node 1
        [2, 0, 1, 2, -1]    # triangle: connect node 0,1,2
    ], device=device, dtype=torch.long)

    measures = compute_node_measures_torch(nodes, elems)
    print("Node measures:\n", measures)

def test_convert_structured_data_torch(features):
    # EXAMPLE USAGE
    # Make some synthetic data
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ndata = 2
    nx, ny = 3, 4
    nfeatures = 1  # e.g. 1 extra feature

    # Suppose coords_list = [coordx, coordy]
    # shape: (ndata, nx, ny)
    coordx = torch.linspace(0, 2, nx).view(1, nx, 1).expand(ndata, nx, ny)
    coordy = torch.linspace(0, 3, ny).view(1, 1, ny).expand(ndata, nx, ny)

    # features shape: (ndata, nx, ny, nfeatures)
    features = features

    coords_list = [coordx, coordy]

    # Convert structured data to unstructured (triangular mesh)
    nodes_list, elems_list, features_list = convert_structured_data_torch(
        coords_list, 
        features, 
        nnodes_per_elem=4,
        feature_include_coords=True,
        device=device
    )

    # Now you can see each element in nodes_list, elems_list, features_list is a GPU Tensor if device="cuda".
    print("nodes_list[0].shape =", nodes_list[0].shape)
    print("elems_list[0].shape =", elems_list[0].shape)
    print("features_list[0].shape =", features_list[0].shape)

####################
# numpy version
####################

def test_compute_node_measures():

    # Example usage:
    # Suppose we have 4 nodes in 2D
    # shape: (4, 2)
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    # Suppose we have two elements:
    # 1) A line (elem_dim=1) connecting node indices 0,1
    # 2) A triangle (elem_dim=2) connecting node indices 0,1,2
    # We set -1 as padding after that when not needed
    elems = np.array([
        [1, 0, 1, -1, -1],  # line: connect node 0 and node 1
        [2, 0, 1, 2, -1]    # triangle: connect node 0,1,2
    ])

    measures = compute_node_measures(nodes, elems)
    print("Node measures:\n", measures)

def test_convert_structured_data(features):

    # EXAMPLE USAGE
    # Make some synthetic data
    ndata = 2
    nx, ny = 3, 4
    nfeatures = 1  # e.g. 1 extra feature

    # Suppose coords_list = [coordx, coordy]
    # shape: (ndata, nx, ny)
    coordx = np.tile(np.linspace(0, 2, nx).reshape(1, nx, 1), (ndata, 1, ny))
    coordy = np.tile(np.linspace(0, 3, ny).reshape(1, 1, ny), (ndata, nx, 1))

    # features shape: (ndata, nx, ny, nfeatures)
    features = features.numpy()

    coords_list = [coordx, coordy]

    # Convert structured data to unstructured (triangular mesh)
    nodes_list, elems_list, features_list = convert_structured_data(
        coords_list, 
        features, 
        nnodes_per_elem=4,
        feature_include_coords=True
    )

    # Now you can see each element in nodes_list, elems_list, features_list is a numpy array.
    print("nodes_list[0].shape =", nodes_list[0].shape)
    print("elems_list[0].shape =", elems_list[0].shape)
    print("features_list[0].shape =", features_list[0].shape)



if __name__ == "__main__":


    ndata = 2
    nx, ny = 3, 4
    nfeatures = 1  # e.g. 1 extra feature
    features = torch.rand(ndata, nx, ny, nfeatures)
    test_convert_structured_data_torch(features)
    test_convert_structured_data(features)

    test_compute_node_measures_torch()
    test_compute_node_measures()

