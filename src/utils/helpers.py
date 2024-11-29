import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR

from src.datasets.dataset import MockDataset
from src.models import PAE


def resolve_dataset_class(name):
    return {
        'mock_dataset': MockDataset,
    }[name]


def resolve_model_class(name, cfg):
    return {
        'pae': PAE(cfg),
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