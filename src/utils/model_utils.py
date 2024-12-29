import torch


def count_model_params(model: torch.nn.Module, only_trainable: bool=True) -> int:
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else: 
        return sum(p.numel() for p in model.parameters())
    