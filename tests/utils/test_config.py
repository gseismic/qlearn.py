import config
import numpy as np
from rlearn.utils.config import Config
import pytest

def test_constructor():
    A = Config()
    A.lr = 0.01
    assert A.lr == 0.01
    A.method = Config()
    A.method.optim = 1
    assert A.method.optim == 1
    A.method.optim = 2
    assert A.method.optim == 2
    
    A.clear()
    assert A.to_dict() == {}
    with pytest.raises(KeyError):
        print(A.lr)
    with pytest.raises(KeyError):
        print(A.method)
    with pytest.raises(KeyError):
        print(A.method.optim)
    
    A = Config.from_dict({'lr': 0.01, 'method': {'optim': 1}})
    assert A.lr == 0.01
    assert A.method.optim == 1
    A.method.optim = 2
    assert A.method.optim == 2
    # for k, v in A.items():
    #     print(k, v)
    
    A = Config({'lr': 0.01, 'method': {'optim': 1}})
    print(A)
    assert A.get_optional('method.optim') == 1
    
    
def test_get_key():
    A = Config.from_dict({'lr': 0.01, 'method': {'optim': 1}})
    print(A)
    lr = A.get('lr')
    assert lr == 0.01
    optim = A.get_required('method.optim')
    assert optim == 1
    ok = A.get('method.ok', 0)
    assert ok == 0
    with pytest.raises(KeyError):
        A.get_required('method.not_exist')
    
    not_exist = A.get_optional('method.not_exist', {'a': 1})
    assert isinstance(not_exist, Config)
    assert not_exist.a == 1
    print(A.to_json())
    
    A.to_json_file('test.json')
    B = Config.from_json_file('test.json')
    assert B.lr == 0.01
    assert B.method.optim == 1
    print(B)

def test_set_key():
    A = Config()
    A['optim.lr'] = 0.001
    assert A['optim.lr'] == 0.001
    assert A.optim.lr == 0.001
    A['algorithm.optim.lr'] = 0.001
    assert A.algorithm.optim.lr == 0.001
    print(A.to_dict())
    
    A.clear()
    with pytest.raises(KeyError):
        y = A.x1.xx1
    with pytest.raises(KeyError):
        A.x1.xx1 = 2

def test_get_required():
    A = Config.from_dict({'lr': 0.01, 'method': {'optim': 1}, 'ok': 1})
    lr = A.get_required('lr', min=0.001, max=0.1)
    assert lr == 0.01
    with pytest.raises(ValueError):
        A.get_required('lr', min=0.011, max=0.1)
    with pytest.raises(ValueError):
        A.get_required('lr', min=0.001, max=0.009)
    
    ok = A.get_required('ok', in_values=[1, 2, 3], is_int=True)
    assert ok == 1
    with pytest.raises(ValueError):
        A.get_required('ok', in_values=[2, 3], is_int=True)
    with pytest.raises(TypeError): 
        A.get_required('ok', is_float=True)
        
def test_from_dict():
    A = Config.from_dict({'lr': 0.01, 'method': {'optim': 1}})
    assert A.lr == 0.01
    assert A.method.optim == 1
    A.ok = 1
    assert A.ok == 1
    A.set_nested('method.optim', 2)
    print(A)
    # print(A.lr)
    # print(A.method)
    # print(A.method.optim)
    print(A.lr)
    print(A.method.optim)

def test_to_yaml():
    A = Config.from_dict({'lr': 0.01, 'method': {'optim': 1}})
    A.to_yaml_file('test.yaml')
    B = Config.from_yaml_file('test.yaml')
    assert B.lr == 0.01
    assert B.method.optim == 1
    print(B)

def test_to_json():
    A = Config.from_dict({'lr': 0.01, 'method': {'optim': 1}})
    A.to_json_file('test.json')
    B = Config.from_json_file('test.json')
    assert B.lr == 0.01
    assert B.method.optim == 1
    print(B)
    
if __name__ == '__main__':
    if 0:
        test_constructor()
    if 0:
        test_get_key()
    if 1:
        test_set_key()
    if 0:
        test_get_required()
    if 0:
        test_from_dict()
    if 0:
        test_to_yaml()
    if 0:
        test_to_json()
