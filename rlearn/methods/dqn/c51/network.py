import torch
import torch.nn as nn
import torch.nn.functional as F 

class C51Network(nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms, v_min, v_max, hidden_layers=[128, 128], activation=nn.ReLU):
        super(C51Network, self).__init__()
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim * num_atoms))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        # state: (batch_size, state_dim)
        distribution = self.network(state) # 
        distribution = distribution.view(-1, self.action_dim, self.num_atoms)
        distribution = nn.functional.softmax(distribution, dim=-1)
        return distribution