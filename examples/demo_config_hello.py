from rlearn.utils import config

cfg = config.from_dict({'method': {'optim': 'sgd', 'lr': 0.001}})
lr = cfg.get_required('method.lr', min=0.001, is_numeric=True)
optim = cfg.get_optional('method.optim', is_str=True, in_values=['sgd', 'adam'])
value = cfg.get_optional('method.value', default=0.001, min=0.001, is_float=True)

print(f'lr: {lr}, optim: {optim}, value: {value}')
# lr: 0.001, optim: sgd, value: 0.001
cfg.to_json_file('demo_config/hello.json')

cfg_loaded = config.from_json_file('demo_config/hello.json')
print(cfg_loaded)
