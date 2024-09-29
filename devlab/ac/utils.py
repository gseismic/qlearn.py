import torch.nn as nn
import torch.optim as optim

ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'leaky_relu': nn.LeakyReLU
}

OPTIMIZERS = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop
}

def get_activation_class(activation, default='relu'):
    if activation is None:
        if isinstance(default, str):
            return ACTIVATIONS[default]
        elif issubclass(default, nn.Module):
            return default
        else:
            raise ValueError(f"Not supported default activation: {default}")
    elif isinstance(activation, str):
        return ACTIVATIONS[activation]
    elif issubclass(activation, nn.Module):
        return activation
    else:
        raise ValueError(f"Not supported activation: {activation}")

def get_optimizer_class(optimizer, default='adam'):
    if optimizer is None:
        if isinstance(default, str):
            return OPTIMIZERS[default]
        elif issubclass(default, optim.Optimizer):
            return default
        else:
            raise ValueError(f"Not supported default optimizer: {default}")
    elif isinstance(optimizer, str):    
        return OPTIMIZERS[optimizer]
    elif issubclass(optimizer, optim.Optimizer):
        return optimizer
    else:
        raise ValueError(f"Not supported optimizer: {optimizer}")