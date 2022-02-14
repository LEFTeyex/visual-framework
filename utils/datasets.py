r"""
Dataset utils.
Consist of Dataset that designed for different tasks.
"""

from torch.utils.data import Dataset


# TODO 2020.2.14 design the class
class DatasetDetect(Dataset):
    def __init__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
