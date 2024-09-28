
from .main.acnet_mlp import ACNet_MLP
from .main.acnet_cnn import ACNet_CNN

# 修改 get_network 函数以包含新的网络类型
def get_acnet(tag, state_dim, action_dim, net_type, policy_net_config, noise_level):
    if tag == 'main':
        params = policy_net_config['params']
        
        if net_type == 'MLP':
            return ACNet_MLP(
                state_dim,
                action_dim,
                params['hidden_dims'],
                params['activation'],
                params['init_type'],
                params['use_noisy'],
                params['factorized'],
                params['rank'],
                params['std_init'],
                params['use_softmax'],
                noise_level
            )
        elif net_type == 'CNN':
            if isinstance(state_dim, (tuple, list)):
                if len(state_dim) == 3:
                    input_channels = state_dim[0]
                elif len(state_dim) == 2:
                    input_channels = 1
                else:
                    raise ValueError("For CNN networks, state_dim should be a tuple or list of length 2 (height, width) for single-channel images or 3 (channels, height, width) for multi-channel images")
            else:
                raise ValueError("For CNN networks, state_dim should be a tuple or list")
            
            return ACNet_CNN(
                input_channels,
                action_dim,
                params['conv_layers'],
                params['hidden_dims'],
                params['activation'],
                params['init_type'],
                params['use_noisy'],
                params['factorized'],
                params['rank'],
                params['std_init'],
                params['use_softmax'],
                noise_level
            )
        else:
            raise ValueError(f"Unsupported network type: {net_type}")
    else:
        raise ValueError(f"Unknown network tag: {tag}")