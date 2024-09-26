import torch.nn as nn

def init_weight(module: nn.Module, 
                weight_init_method: str = 'kaiming', 
                weight_init_kwargs = None,
                bias_init_method: str = 'constant',
                bias_init_kwargs = None):
    # weight default kwargs
    if weight_init_method == 'kaiming_uniform':
        weight_init_kwargs = {'a': 0.0, 'mode': 'fan_in', 'nonlinearity': 'relu'}
    elif weight_init_method == 'kaiming_normal':
        weight_init_kwargs = {'a': 0.0, 'mode': 'fan_in', 'nonlinearity': 'relu'}
    elif weight_init_method == 'xavier_uniform':
        weight_init_kwargs = {'gain': 1.0}
    elif weight_init_method == 'xavier_normal':
        weight_init_kwargs = {'gain': 1.0}
    elif weight_init_method == 'uniform':
        weight_init_kwargs = {'a': 0.0, 'b': 1.0}
    elif weight_init_method == 'normal':
        weight_init_kwargs = {'mean': 0.0, 'std': 1.0}
    elif weight_init_method == 'constant':
        weight_init_kwargs = {'val': 0.0}
    else:
        raise ValueError(f'Invalid weight initialization method: {weight_init_method}')
    
    # bias default kwargs
    if bias_init_method == 'constant':
        bias_init_kwargs = {'val': 0.0}
    elif bias_init_method == 'normal':
        bias_init_kwargs = {'mean': 0.0, 'std': 1.0}
    elif bias_init_method == 'uniform':
        bias_init_kwargs = {'a': 0.0, 'b': 1.0}
    else:
        raise ValueError(f'Invalid bias initialization method: {bias_init_method}') 
    
    if weight_init_method == 'xavier_uniform':
        nn.init.xavier_uniform_(module.weight, **weight_init_kwargs)
    elif weight_init_method == 'xavier_normal':
        nn.init.xavier_normal_(module.weight, **weight_init_kwargs)
    elif weight_init_method == 'kaiming_uniform':
        nn.init.kaiming_uniform_(module.weight, **weight_init_kwargs)
    elif weight_init_method == 'kaiming_normal':
        nn.init.kaiming_normal_(module.weight, **weight_init_kwargs)
    elif weight_init_method == 'uniform':
        nn.init.uniform_(module.weight, **weight_init_kwargs)
    elif weight_init_method == 'normal':
        nn.init.normal_(module.weight, **weight_init_kwargs)
    elif weight_init_method == 'constant':
        nn.init.constant_(module.weight, **weight_init_kwargs)
    else:
        raise ValueError(f'Invalid weight initialization method: {weight_init_method}')
    
    if module.bias is not None:
        if bias_init_method == 'constant':
            nn.init.constant_(module.bias, **bias_init_kwargs)
        elif bias_init_method == 'normal':
            nn.init.normal_(module.bias, **bias_init_kwargs)
        elif bias_init_method == 'uniform':
            nn.init.uniform_(module.bias, **bias_init_kwargs)
        else:
            raise ValueError(f'Invalid bias initialization method: {bias_init_method}')

def init_linear_weight(module: nn.Module, 
                       weight_init_method: str = 'kaiming', 
                       weight_init_kwargs: dict = {},
                       bias_init: float = 0.0, 
                       bias_init_method: str = 'constant',
                       bias_init_kwargs: dict = {}):
    if not isinstance(module, nn.Linear):
        return
    return init_weight(module, weight_init_method, weight_init_kwargs, bias_init, bias_init_method, bias_init_kwargs)


def init_conv2d_weight(module: nn.Module, 
                       weight_init_method: str = 'kaiming', 
                       weight_init_kwargs: dict = {},
                       bias_init: float = 0.0, 
                       bias_init_method: str = 'constant',
                       bias_init_kwargs: dict = {}):
    if not isinstance(module, nn.Conv2d):
        return
    return init_weight(module, weight_init_method, weight_init_kwargs, bias_init, bias_init_method, bias_init_kwargs)

def init_batch_norm_weight(module: nn.Module, 
                       weight_init_method: str = 'uniform', 
                       weight_init_kwargs: dict = {},
                       bias_init: float = 0.0, 
                       bias_init_method: str = 'constant',
                       bias_init_kwargs: dict = {}):
    if not isinstance(module, nn.BatchNorm2d):
        return
    return init_weight(module, weight_init_method, weight_init_kwargs, bias_init, bias_init_method, bias_init_kwargs)

def batch_norm_weight_init(module: nn.Module, 
                       weight_init_method: str = 'uniform', 
                       weight_init_kwargs: dict = {},
                       bias_init: float = 0.0, 
                       bias_init_method: str = 'constant',
                       bias_init_kwargs: dict = {}):
    if not isinstance(module, nn.BatchNorm2d):
        return
    return init_weight(module, weight_init_method, weight_init_kwargs, bias_init, bias_init_method, bias_init_kwargs)