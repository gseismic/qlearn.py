import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from ....cnn.main.cnn import CNN
from ....mlp.main.mlp import MLP

class ACNet_CNN(nn.Module):
    def __init__(self, input_channels: int, action_dim: int, conv_layers: List[Tuple[int, int, int]], hidden_dims: List[int], 
                 activation: str = 'relu', init_type: str = 'kaiming', use_noisy: bool = False, factorized: bool = True, 
                 rank: int = 0, std_init: float = 0.4, use_softmax: bool = True):
        super(ACNet_CNN, self).__init__()
        
        self.feature_extractor = CNN(input_channels, conv_layers, activation, init_type)
        
        self.feature_dim: Optional[int] = None
        
        self.actor: Optional[MLP] = None
        self.critic: Optional[MLP] = None
        
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.init_type = init_type
        self.use_noisy = use_noisy
        self.factorized = factorized
        self.rank = rank
        self.std_init = std_init
        self.use_softmax = use_softmax
        self.action_dim = action_dim

    def _initialize_networks(self, input_shape: Tuple[int, ...]) -> None:
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            sample_output = self.feature_extractor(sample_input)
            self.feature_dim = sample_output.numel() // sample_output.size(0)
        
        self.actor = MLP(
            input_dim=self.feature_dim,
            output_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            init_type=self.init_type,
            use_noisy=self.use_noisy,
            factorized=self.factorized,
            rank=self.rank,
            std_init=self.std_init,
            output_activation='softmax' if self.use_softmax else None
        )

        self.critic = MLP(
            input_dim=self.feature_dim,
            output_dim=1,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            init_type=self.init_type,
            use_noisy=self.use_noisy,
            factorized=self.factorized,
            rank=self.rank,
            std_init=self.std_init)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.actor is None or self.critic is None:
            self._initialize_networks(state.shape[1:])
        
        features = self.feature_extractor(state)
        features = features.view(features.size(0), -1)
        return self.actor(features), self.critic(features)

    def reset_noise(self) -> None:
        if self.actor is not None and self.critic is not None:
            self.actor.reset_noise()
            self.critic.reset_noise()

    def set_noise_scale(self, scale: float) -> None:
        if self.actor is not None and self.critic is not None:
            self.actor.set_noise_scale(scale)
            self.critic.set_noise_scale(scale)
            
__all__ = ['ACNet_CNN']