import math
import random
from typing import Iterator, Optional, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


def create_duplicate_dataset(DatasetBaseClass):
    class DupDataset(DatasetBaseClass):
        def __init__(self, copy, **kwargs):
            super().__init__(**kwargs)

            self.copy = copy
            self.length = super().__len__()

        def __len__(self):
            return self.copy * self.length

        def __getitem__(self, index):
            true_index = index % self.length
            return super().__getitem__(true_index)

        def get_img_info(self, index):
            true_index = index % self.length
            return super().get_img_info(true_index)

    return DupDataset
