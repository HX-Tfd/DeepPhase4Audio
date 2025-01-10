import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
import yaml

from src.models.PAE import AE, PAE, PAEInputFlattened
from src.datasets.data_processing import AudioDataset
from src.datasets.dataset import MockDataset
from src.models.VQ_PAE import VQ_AE


class DotDict:
    """
        A class to conveniently wrap dictionaries to access them by dot
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)
            self.__dict__[key] = value

    def __getitem__(self, item):
        return self.__dict__[item]

    def __repr__(self):
        return str(self.__dict__)
    
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"'DotDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = DotDict(value)
        self.__dict__[name] = value
    


def replace_value(d: DotDict, ref, value):
    """
    Replaces the value in the DotDict instance by directly passing the reference to the sub-object.
    
    :param d: The root DotDict instance
    :param ref: A reference to the specific value to replace (e.g., d.me)
    :param value: The new value to set
    :return: The modified DotDict
    """
    def find_parent_and_key(current, target):
        """Recursively find the parent DotDict and key of the target value."""
        for key, val in current.__dict__.items():
            if val is target:  
                return current, key
            if isinstance(val, DotDict):
                result = find_parent_and_key(val, target)
                if result:
                    return result
        return None

    result = find_parent_and_key(d, ref)
    if not result:
        raise ValueError("Reference not found in the DotDict.")
    
    parent, key = result
    parent[key] = value
    return d


def resolve_dataset_class(name):
    return {
        'mock_dataset': MockDataset,
        'audio_dataset': AudioDataset,
    }[name]


def resolve_model_class(name, cfg):
    return {
        'pae': PAE(cfg),
        'pae_flat': PAEInputFlattened(cfg),
        # 'ae': AE(cfg), 
        'vq_pae': VQ_AE(cfg)
    }[name]


def resolve_optimizer(cfg, params):
    if cfg.optimizer == 'sgd':
        return SGD(
            params,
            lr=cfg.optimizer_lr,
            momentum=cfg.optimizer_momentum,
            weight_decay=cfg.optimizer_weight_decay,
        )
    elif cfg.optimizer == 'adam':
        return Adam(
            params,
            lr=cfg.optimizer_lr,
            weight_decay=cfg.optimizer_weight_decay,
        )
    else:
        raise NotImplementedError


def resolve_lr_scheduler(cfg, optimizer):
    if cfg.lr_scheduler == 'poly':
        return LambdaLR(
            optimizer,
            lambda ep: max(1e-6, (1 - ep / cfg.num_epochs) ** cfg.lr_scheduler_power)
        )
    else:
        print(f"scheduler {cfg.lr_scheduler} is not found or not implemented yet!")
        raise NotImplementedError
    

def get_device_accelerator(preferred_accelerator='auto'):
    """
    Determines the best device accelerator for PyTorch Lightning based on availability.
    
    Args:
        preferred_accelerator (str): The preferred device accelerator. Options: 'auto', 'cpu', 'cuda', 'mps', 'tpu'.
                                     Defaults to 'auto', which selects the best available device.
    
    Returns:
        str: The selected device accelerator.
    """
    # Handle 'auto' by selecting the best available accelerator
    if preferred_accelerator == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():  # For macOS Metal (Apple Silicon)
            return 'mps'
        elif torch.distributed.is_torchelastic_launched() and torch.distributed.is_available():
            return 'tpu'
        else:
            return 'cpu'
    
    # Validate specific user-provided accelerators
    available_accelerators = {
        'cpu': True,
        'cuda': torch.cuda.is_available(),
        'mps': torch.backends.mps.is_available(),
        'tpu': torch.distributed.is_torchelastic_launched() and torch.distributed.is_available(),
    }
    
    if preferred_accelerator in available_accelerators and available_accelerators[preferred_accelerator]:
        return preferred_accelerator
    else:
        raise ValueError(
            f"Preferred accelerator '{preferred_accelerator}' is not available. "
            f"Available options: {[k for k, v in available_accelerators.items() if v]}."
        )
        
def flatten_dict(d, absolute=False, parent_key='', separator='.'):
    """
    Flatten a nested dictionary, keeping only the inner (leaf) values.
    
    If absolute = True, the keys will be the absolute access path, otherwise only the key of the leaf
    """
    items = []
    for k, v in d.items():
        if absolute:
            new_key = f"{parent_key}{separator}{k}" if parent_key else k
        else:
            new_key = f"{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))
    return dict(items)
