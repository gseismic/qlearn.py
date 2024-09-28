from .linear import get_linear
from .mlp import get_mlp

block_dict = {
    'linear': get_linear,
    'mlp': get_mlp
}

def get_block(name: str, tag, *args, **kwargs):
    try:
        return block_dict[name](tag, *args, **kwargs)
    except KeyError:
        raise ValueError(f"Unknown module name: {name}")