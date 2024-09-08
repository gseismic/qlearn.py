import config
import numpy as np
from rlearn.spaces import Discrete, MultiDiscrete, Box
import torch
import torch.nn as nn

def test_box():
    space = Box(low=0, high=1, shape=(2,))
    assert space.contains(np.array([0.5, 0.5]))
    assert not space.contains(np.array([1.5, 0.5]))
    
    space = Box(low=[0, 0], high=[1, 2], shape=(2,))
    for i in range(10):
        x = space.sample()
        assert space.contains(x)
        assert np.all(x >= space.low)
        assert np.all(x <= space.high)
        assert x[0] >= 0 and x[0] <= 1
        assert x[1] >= 0 and x[1] <= 2

def test_discrete():
    space = Discrete(n=5)
    for i in range(10):
        x = space.sample()
        assert space.contains(x)
        assert x >= 0 and x < space.n

def test_multidiscrete():
    space = MultiDiscrete(nvec=[2, 3, 4])
    for i in range(10):
        x = space.sample()
        assert space.contains(x)
        assert np.all(x >= 0)
        assert np.all(x < space.nvec)   
        
    assert space.contains(np.array([1, 2, 3]))
    assert not space.contains(np.array([2, 3, 4]))
    assert not space.contains(np.array([3, 4, 5]))
    assert not space.contains(np.array([-1, 2, 3]))
    
if __name__ == '__main__':
    if 1:
        test_box()
    if 1:
        test_discrete()
    if 1:
        test_multidiscrete()
