import torch
import torch.nn as nn
from typing import List, Tuple
from ....mlp.main.mlp import MLP

class ACNet_MLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int], 
                 activation: str = 'relu', init_type: str = 'kaiming', 
                 use_noisy: bool = False, factorized: bool = True, 
                 rank: int = 0, std_init: float = 0.4, use_softmax: bool = True):
        super(ACNet_MLP, self).__init__()
        
        self.actor = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            init_type=init_type,
            use_noisy=use_noisy,
            factorized=factorized,
            rank=rank,
            std_init=std_init,
            output_activation='softmax' if use_softmax else None
        )

        self.critic = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation,
            init_type=init_type,
            use_noisy=use_noisy,
            factorized=factorized,
            rank=rank,
            std_init=std_init
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(state), self.critic(state)

    def reset_noise(self) -> None:
        self.actor.reset_noise()
        self.critic.reset_noise()

    def set_noise_scale(self, scale: float) -> None:
        self.actor.set_noise_scale(scale)
        self.critic.set_noise_scale(scale)

__all__ = ['ACNet_MLP']