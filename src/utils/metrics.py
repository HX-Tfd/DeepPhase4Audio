import torch
import torch.nn as nn

def simple_metric(x_true, x_pred):
    """computes the Mean Absolute Error"""
    return nn.L1Loss()(x_true, x_pred)
