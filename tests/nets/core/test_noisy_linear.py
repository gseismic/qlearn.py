import pytest
import torch
from rlearn.nets.core.noisy_linear import DenseNoisyLinear, FactorizedNoisyLinear

@pytest.fixture
def layer_params():
    return {
        'in_features': 10,
        'out_features': 5,
        'batch_size': 32
    }

def test_dense_noisy_linear_shape(layer_params):
    layer = DenseNoisyLinear(layer_params['in_features'], layer_params['out_features'])
    input_tensor = torch.randn(layer_params['batch_size'], layer_params['in_features'])
    output = layer(input_tensor)
    assert output.shape == (layer_params['batch_size'], layer_params['out_features'])

def test_factorized_noisy_linear_shape(layer_params):
    layer = FactorizedNoisyLinear(layer_params['in_features'], layer_params['out_features'], k=3)
    input_tensor = torch.randn(layer_params['batch_size'], layer_params['in_features'])
    output = layer(input_tensor)
    assert output.shape == (layer_params['batch_size'], layer_params['out_features'])

def test_noisy_linear_noise_generation(layer_params):
    layer = DenseNoisyLinear(layer_params['in_features'], layer_params['out_features'])
    initial_noise = layer.weight_epsilon.clone()
    layer.reset_noise()
    assert not torch.allclose(initial_noise, layer.weight_epsilon)

def test_factorized_noisy_linear_noise_generation(layer_params):
    layer = FactorizedNoisyLinear(layer_params['in_features'], layer_params['out_features'], k=3)
    initial_noise = layer.weight_epsilon.clone()
    layer.reset_noise()
    assert not torch.allclose(initial_noise, layer.weight_epsilon)

def test_exploration_factor(layer_params):
    layer = DenseNoisyLinear(layer_params['in_features'], layer_params['out_features'], exploration_factor=0.5)
    input_tensor = torch.randn(layer_params['batch_size'], layer_params['in_features'])
    output1 = layer(input_tensor)
    layer.exploration_factor = 1.0
    output2 = layer(input_tensor)
    assert not torch.allclose(output1, output2)

def test_training_vs_eval_mode(layer_params):
    layer = DenseNoisyLinear(layer_params['in_features'], layer_params['out_features'])
    input_tensor = torch.randn(layer_params['batch_size'], layer_params['in_features'])
    
    layer.train()
    output_train = layer(input_tensor)
    
    layer.eval()
    output_eval = layer(input_tensor)
    
    assert not torch.allclose(output_train, output_eval)

def test_factorised_noisy_linear_k_parameter(layer_params):
    k_values = [1, 3, 5]
    for k in k_values:
        layer = FactorizedNoisyLinear(layer_params['in_features'], layer_params['out_features'], k=k)
        assert layer.k == k
        assert layer.epsilon_in.shape == (k, layer_params['in_features'])
        assert layer.epsilon_out.shape == (k, layer_params['out_features'])

def test_init_methods(layer_params):
    init_methods = ['uniform', 'xavier', 'kaiming']
    for method in init_methods:
        layer = DenseNoisyLinear(layer_params['in_features'], layer_params['out_features'], init_method=method)
        assert layer.init_method == method

    with pytest.raises(ValueError):
        DenseNoisyLinear(layer_params['in_features'], layer_params['out_features'], init_method='invalid_method')

def test_factorized_noisy_linear_k_parameter(layer_params):
    k_values = [1, 3, 5]
    for k in k_values:
        layer = FactorizedNoisyLinear(layer_params['in_features'], layer_params['out_features'], k=k)
        assert layer.k == k
        assert layer.epsilon_in.shape == (k, layer_params['in_features'])
        assert layer.epsilon_out.shape == (k, layer_params['out_features'])
