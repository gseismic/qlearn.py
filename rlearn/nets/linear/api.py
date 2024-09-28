import torch.nn as nn
from .main.noisy_linear import NoisyLinear

def get_linear(tag, in_features: int, out_features: int, noisy: bool = False, 
               factorized: bool = True, rank: int = 0, std_init: float = 0.4) -> nn.Module:
    if tag == 'main':
        if noisy:
            return NoisyLinear(in_features, out_features, std_init=std_init, factorized=factorized, rank=rank)
        else:
            return nn.Linear(in_features, out_features)
    else:
        raise ValueError(f"Unknown tag: {tag}")

# def get_linear(in_features: int, out_features: int, noisy: bool = False, 
#                factorized: bool = True, rank: int = 0, std_init: float = 0.4) -> nn.Module:
#     if noisy:
#         return NoisyLinear(in_features, out_features, std_init=std_init, factorized=factorized, rank=rank)
#     else:
#         return nn.Linear(in_features, out_features)

def is_linear(block: nn.Module) -> bool:
    return isinstance(block, (nn.Linear, NoisyLinear))

def is_noisy_linear(block: nn.Module) -> bool:
    return isinstance(block, NoisyLinear)

__all__ = ['get_linear', 'is_linear', 'is_noisy_linear']
