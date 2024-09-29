import torch.optim as optim

def get_optimizer(parameters, optimizer_config):
    optimizer_type = optimizer_config['type']
    optimizer_params = optimizer_config['params']

    try:
        optimizer_class = getattr(optim, optimizer_type)
        return optimizer_class(parameters, **optimizer_params)
    except AttributeError:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")