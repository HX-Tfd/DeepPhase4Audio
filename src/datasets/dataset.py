# A mock dataset for testing dry run
import torch

from src.datasets.definitions import SPLIT_TRAIN, SPLIT_TEST, SPLIT_VALID


class MockDataset(torch.utils.data.Dataset):
    """
    This mock dataset just returns random tensors of the input shape
    We give it a mock length of 1000 (800 train, 100 val, 100 test)
    """
    def __init__(self, dataset_root, split, shape):
        assert split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST), f'Invalid split {split}'
        self.dataset_root = dataset_root
        self.split = split
        self.transforms = None
        self.shape = shape


    def __getitem__(self, index):
        return torch.randn(self.shape)


    def __len__(self):
        return {
            SPLIT_TRAIN: 800, 
            SPLIT_VALID: 100, 
            SPLIT_TEST: 100
        }[self.split]


    @staticmethod
    def custom_static_method():
        # add static methods
        pass


    @property
    def dataset_prop():
        # add dataset props 
        pass