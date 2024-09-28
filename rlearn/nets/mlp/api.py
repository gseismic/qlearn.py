from .main.mlp import MLP

def get_mlp(tag='main', *args, **kwargs):
    if tag == 'main':
        return MLP(*args, **kwargs)
    else:
        raise ValueError(f"Unknown mlp type: {tag}")


__all__ = ['get_mlp']