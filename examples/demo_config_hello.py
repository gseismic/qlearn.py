from rlearn.utils import config

cfg = config.from_dict({'method': {'optim': 'sgd', 'lr': 0.001}})
lr = cfg.get_required('method.lr', min=0.001, is_numeric=True)
optim = cfg.get_optional('method.optim', is_str=True, in_values=['sgd', 'adam'])
value = cfg.get_optional('method.value', default=0.001, min=0.001, is_float=True)

cfg.set('method.v_min', 3.0, is_float=True, gt=0) # gt: greater than
cfg.set('method.v_max', 5.0, is_float=True, gt='method.v_min') # gt: greater than

print(f'lr: {lr}, optim: {optim}, value: {value}')
# lr: 0.001, optim: sgd, value: 0.001
cfg.to_json_file('demo_config/hello.json')
cfg_loaded = config.from_json_file('demo_config/hello.json')
# print(cfg_loaded.to_dict())
