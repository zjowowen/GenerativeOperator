import random
from typing import List, Dict

import h5py
import numpy as np
import torch

import numpy as np
import torch

from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, LazyMemmapStorage


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        max_size: int = 100000,
        folder_path="data",
    ):
        self.key_list = []
        self.params = {}
        self.len = 0
        self.storage = LazyMemmapStorage(max_size=max_size)
        self.dataset_names = []
        self.load_data()
        self.get_min_max()

    def load_data(
        self
    ):
        
        pass

    def get_min_max(self):
        all_data = self.storage.get(range(self.len))["gt"].reshape(-1, 2)

        xmin = all_data.min(0)[0][0]
        xmax = all_data.max(0)[0][0]
        ymin = all_data.min(0)[0][1]
        ymax = all_data.max(0)[0][1]

        self.min, self.max = torch.tensor((xmin, ymin)), torch.tensor((xmax, ymax))
        return self.min, self.max

    def extend_data(self, data: Dict):
        # keys = ["gt", "params"]

        len_after_extend = self.len + 1

        self.storage.set(
            range(self.len, len_after_extend),
            TensorDict(
                data,
                batch_size=[1],
            ),
        )
        self.len = len_after_extend

    def __getitem__(self, index):
        data = self.storage.get(index=index)
        return data

    def __len__(self):
        return self.len
