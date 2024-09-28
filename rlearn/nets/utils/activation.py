import torch.nn as nn
from typing import Dict

def get_activation(activation_name: str) -> nn.Module:
    activations: Dict[str, nn.Module] = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'prelu': nn.PReLU(),
        'celu': nn.CELU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'softmax': nn.Softmax(dim=-1),
        'softplus': nn.Softplus(),
        'softsign': nn.Softsign(),
        'hardtanh': nn.Hardtanh(),
        'mish': nn.Mish(),
        'swish': nn.SiLU(),
        'hardswish': nn.Hardswish(),
        'log_softmax': nn.LogSoftmax(dim=-1),
    }
    try:
        return activations[activation_name.lower()]
    except KeyError:
        raise ValueError(f"Unknown activation function: {activation_name}")


__all__ = ['get_activation']