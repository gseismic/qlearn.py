import torch.nn as nn
from typing import Dict, Callable

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

def get_initializer(init_name: str) -> Callable:
    initializers: Dict[str, Callable] = {
        'xavier_uniform': nn.init.xavier_uniform_,
        'xavier_normal': nn.init.xavier_normal_,
        'kaiming_uniform': nn.init.kaiming_uniform_,
        'kaiming_normal': nn.init.kaiming_normal_,
        'orthogonal': nn.init.orthogonal_,
    }
    try:
        return initializers[init_name.lower()]
    except KeyError:
        raise ValueError(f"Unknown initializer: {init_name}")


__all__ = ['get_activation', 'get_initializer']