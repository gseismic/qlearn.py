import torch.nn as nn
import numpy as np

class Linear(nn.Linear):
    """
    Linear 类是一个带有可选初始化方法的线性层。| Linear class is a linear layer with optional initialization methods.
    这个类允许在创建时指定权重和偏置的初始化方法。| This class allows specifying the initialization method for weights and biases when created.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 init_method: str = 'kaiming'):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.init_method = init_method
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.init_method == 'uniform':
            limit = 1 / np.sqrt(self.in_features)
            nn.init.uniform_(self.weight, -limit, limit)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -limit, limit)
        elif self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)
        elif self.init_method == 'kaiming':
            nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")