import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_sizes, 
                 hidden_activation=nn.ReLU(), 
                 output_activation=None, 
                 use_batch_norm=False):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_batch_norm = use_batch_norm
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(hidden_activation)
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        if output_activation is not None:
            layers.append(output_activation)
        
        self.network = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

