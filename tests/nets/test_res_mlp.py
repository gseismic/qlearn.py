import config
import numpy as np
from rlearn.nets import ResMLP
import torch
import torch.nn as nn


def test_res_mlp_basic():
    input_size = 10
    hidden_sizes = [32, 32, 16]
    output_size = 5
    
    model = ResMLP(input_size=input_size,
                   output_size=output_size,
                   hidden_sizes=hidden_sizes,
                   hidden_activation=nn.ReLU(),
                   output_activation=nn.Softmax(dim=-1),
                   use_batch_norm=True)
    
    print(model)
    
    # 测试前向传播
    batch_size = 32
    x = torch.randn(batch_size, input_size)  # 批大小为32
    output = model(x)
    assert output.shape == (batch_size, output_size)
    print(f"Output shape: {output.shape}")


if __name__ == '__main__':
    if 1:
        test_res_mlp_basic()
