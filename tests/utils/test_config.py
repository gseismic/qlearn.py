import config
import numpy as np
from rlearn.utils.config import Config
import pytest

def test_constructor():
    cfg = Config()
    cfg.lr = 0.01
    assert cfg.lr == 0.01
    cfg.method = Config()
    cfg.method.optim = 1
    assert cfg.method.optim == 1
    cfg.method.optim = 2
    assert cfg.method.optim == 2
    
    cfg.clear()
    assert cfg.to_dict() == {}
    with pytest.raises(KeyError):
        print(cfg.lr)
    with pytest.raises(KeyError):
        print(cfg.method)
    with pytest.raises(KeyError):
        print(cfg.method.optim)
    
    cfg = Config.from_dict({'lr': 0.01, 'method': {'optim': 1}})
    assert cfg.lr == 0.01
    assert cfg.method.optim == 1
    cfg.method.optim = 2
    assert cfg.method.optim == 2
    # for k, v in cfg.items():
    #     print(k, v)
    
    cfg = Config({'lr': 0.01, 'method': {'optim': 1}})
    print(cfg)
    assert cfg.get_optional('method.optim') == 1
    
    
def test_get_key():
    cfg = Config.from_dict({'lr': 0.01, 'method': {'optim': 1}})
    print(cfg)
    lr = cfg.get('lr')
    assert lr == 0.01
    optim = cfg.get_required('method.optim')
    assert optim == 1
    ok = cfg.get('method.ok', 0)
    assert ok == 0
    with pytest.raises(KeyError):
        cfg.get_required('method.not_exist')
    
    not_exist = cfg.get_optional('method.not_exist', {'a': 1})
    assert isinstance(not_exist, Config)
    assert not_exist.a == 1
    print(cfg.to_json())
    
    cfg.to_json_file('test.json')
    B = Config.from_json_file('test.json')
    assert B.lr == 0.01
    assert B.method.optim == 1
    print(B)

def test_set_key():
    cfg = Config()
    cfg['optim.lr'] = 0.001
    assert cfg['optim.lr'] == 0.001
    assert cfg.optim.lr == 0.001
    cfg['algorithm.optim.lr'] = 0.001
    assert cfg.algorithm.optim.lr == 0.001
    print(cfg.to_dict())
    
    cfg.clear()
    with pytest.raises(KeyError):
        y = cfg.x1.xx1
    with pytest.raises(KeyError):
        cfg.x1.xx1 = 2
    
    cfg.set('optim.lr', 0.001, is_float=True, ge=0)
    assert cfg.optim.lr == 0.001
    with pytest.raises(ValueError):
        cfg.set('optim.lr', 0.001, is_float=True, ge=0.1)
    
def test_get_required():
    cfg = Config.from_dict({'lr': 0.01, 'method': {'optim': 1}, 'ok': 1})
    lr = cfg.get_required('lr', min=0.001, max=0.1)
    assert lr == 0.01
    with pytest.raises(ValueError):
        cfg.get_required('lr', min=0.011, max=0.1)
    with pytest.raises(ValueError):
        cfg.get_required('lr', min=0.001, max=0.009)
    
    ok = cfg.get_required('ok', in_values=[1, 2, 3], is_int=True)
    assert ok == 1
    with pytest.raises(ValueError):
        cfg.get_required('ok', in_values=[2, 3], is_int=True)
    with pytest.raises(TypeError): 
        cfg.get_required('ok', is_float=True)
        
def test_from_dict():
    cfg = Config.from_dict({'lr': 0.01, 'method': {'optim': 1}})
    assert cfg.lr == 0.01
    assert cfg.method.optim == 1
    cfg.ok = 1
    assert cfg.ok == 1
    cfg.set('method.optim', 2)
    print(cfg)
    # print(cfg.lr)
    # print(cfg.method)
    # print(cfg.method.optim)
    print(cfg.lr)
    print(cfg.method.optim)

def test_to_yaml():
    cfg = Config.from_dict({'lr': 0.01, 'method': {'optim': 1}})
    cfg.to_yaml_file('test.yaml')
    B = Config.from_yaml_file('test.yaml')
    assert B.lr == 0.01
    assert B.method.optim == 1
    print(B)

def test_to_json():
    cfg = Config.from_dict({'lr': 0.01, 'method': {'optim': 1}})
    cfg.to_json_file('test.json')
    B = Config.from_json_file('test.json')
    assert B.lr == 0.01
    assert B.method.optim == 1
    print(B)

def test_check_value():
    cfg = Config.from_dict({'lr': 0.01, 'method': {'optim': 1}})
    cfg.set('min_lr', 0.01, ge=0.01)
    with pytest.raises(ValueError):
        cfg.set('max_lr', 0.001, gt='min_lr')
        
    # cfg.clear()
    # cfg.set('max_lr', 0.06, gt='min_lr')
    # cfg.set('v_min', 3.0, is_float=True, gt=0) # gt: greater than
    # cfg.set('v_max', 5.0, is_float=True, gt='v_min') # gt: greater than
    cfg.set('method.v_min', 3.0, is_float=True, gt=0) # gt: greater than
    cfg.set('method.v_max', 5.0, is_float=True, gt='method.v_min') # gt: greater than
    assert cfg.method.v_max == 5.0
    
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
    if 1:
        test_check_value()
