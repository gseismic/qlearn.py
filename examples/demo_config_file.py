from rlearn.utils import config
from rlearn.utils.config import (
    Config, get_required_config, get_optional_config 
)
# 你可能感觉import慢，是因为rlearn导入了torch，并非config模块慢
# | You may feel that the import is slow because rlearn imports torch, not the config module slow
"""A class for managing configuration settings. | 用于管理配置设置的类。

allowed types: 
    - int
        - min <=> ge
        - max <=> le
        - gt
        - ge
        - lt
        - le
        - in_values
    - float
        - min
        - max
        - gt
        - ge
        - lt
        - le
        - in_values
    - str
        - min_length
        - max_length
        - in_values
    - list/tuple
        - min_length
        - max_length
        - element_type
        - element_values
    - dict
        - key_type
        - value_type
    - bool
        - is_bool
    - None
        - is_none
example:
    - config.get_optional('key', default=None, is_numeric=True, gt=0)
    - config.get_required('key', is_str=True, min_length=1)
    - config.get_required('key', is_list=True, min_length=1, element_type=int)
    - config.get_optional('key', default=None, is_dict=True, key_type=str, value_type=int)
    - config.get_required('key', is_bool=True)
    - config.get_required('key', is_none=True)
    - config.get_optional('key', default=None, is_numeric=True, gt=0)
    - config.get_required('key', is_str=True, min_length=1)
    - config.get_required('key', is_list=True, min_length=1, element_type=int)
    - config.get_optional('key', default=None, is_dict=True, key_type=str, value_type=int)
    - config.get_required('key', is_bool=True)
    - config.get_required('key', is_none=True)
    - config.to_json_file('path/to/config.json')
    - config.from_json_file('path/to/config.json')
    - config.to_yaml_file('path/to/config.yaml')
    - config.from_yaml_file('path/to/config.yaml')
"""
#========================================================================
# usage1:
from loguru import logger
config.set_logger(logger)
cfg = config.from_dict({'method': {'optim': 1, 'lr': 0.001}})
cfg.to_json_file('demo_config/demo.json')
cfg.to_yaml_file('demo_config/demo.yaml')
#========================================================================
# usage2:
from rlearn.logger import sys_logger
config.set_logger(sys_logger)
cfg = config.from_dict({'method': {'optim': 1, 'lr': 0.001}})
cfg = config.make_config({'method': {'optim': 1, 'lr': 0.001}})
cfg = config.from_json_file('demo_config/demo.json')
cfg = config.from_yaml_file('demo_config/demo.yaml')
#========================================================================
# usage3:
cfg = Config.from_json_file('demo_config/demo.json')
# lr = cfg.get_optional('method.lr', min=0.001, is_float=True)
lr = cfg.get_optional('method.lr', min=0.001, is_numeric=True)
optim = cfg.get_optional('method.optim', is_int=True, in_values=[1, 2, 3])
not_exist = cfg.get_optional('method.not_exist', default=1, is_int=True, in_values=[1, 2, 3])

optimizer = cfg.get_optional('method.optimizer', 'sgd', in_values=['adam', 'sgd'], is_str=True)

cfg.clear()
cfg['method.value'] = 0.001
assert cfg.get_required('method.value', min=0.001, is_float=True) == 0.001
assert cfg['method.value'] == 0.001
assert cfg['method']['value'] == 0.001

cfg.update({'method': {'optim': 1, 'lr': 0.003}})
print(cfg.to_dict())

value = get_required_config(cfg,'method.value', min=0.001, is_float=True)
#========================================================================
# more usage:
cfg = Config({'method': {'optim': 1, 'lr': 0.001}})
value = get_optional_config(cfg,'method.default', default=0.001, ge=0, le=1, is_float=True)
# 因为不存在，使用的是默认数值，输出log | Because it does not exist, the default value is used, and the log is output
# 2024-09-08 14:31:35 | INFO | Key `method.not_exist` not found, using default value `1`
# 2024-09-08 14:31:35 | INFO | Key `method.default` not found, using default value `0.001`
assert value == 0.001