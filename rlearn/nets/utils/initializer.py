import torch.nn as nn
from typing import Callable, Dict

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

__all__ = ['get_initializer']