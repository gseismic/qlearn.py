# rlearn.py
Reinforcement Learning Library
强化学习库[在开发中 Developing]

## Reference
- 《Math Foundation of Reinforcement Learning》by [Shiyu Zhao]
- 《深度强化学习》by Wang Shuseng 王树森
- gym 
- stable-baselines3
- more ..

## Methods
- [x] StateTableAgent   | Value-Iteration Method
- [x] PolicyTableAgent  | Policy-Iteration Method (with Truncate option)
- [x] QTableAgent       | Tablar-Q-learning Method [on-policy]
- [x] SarsaTableAgent   | SARSA Method [on-policy]
- [ ] Q-learning[off-policy]
- [ ] DQN
- [ ] dual-DQN
- [ ] DDPG
- [ ] HER
- [ ] Monte (REINFORCE)
- [ ] TRPO
- [ ] PPO
- [ ] A2C
- [ ] A3C
- [ ] QAC
- [ ] TD3
- [ ] SAC
- [ ] [REDQ](https://arxiv.org/abs/2101.05982)
- [ ] [DroQ](https://github.com/TakuyaHiraoka/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning)

## TODOs
- [ ] QTableAgent.learn(): return rewards-list

## Examples 例子
Grid-World Demo
Use State-Value-Agent (State Value Iteration Method)
```
start: (0,0)
target: (5,8)
```
| Image 1 | Image 2 |
|---------|---------|
|![state-values](examples/images/grid_world_state_values.png) | ![policy](examples/images/grid_world_policy.gif)|

```
# examples/demo_state_table_agent.py

import os
import torch
from rlearn.envs import grid_world
from rlearn.utils.seed import seed_all
from rlearn.methods.table import StateTableAgent
import matplotlib.pyplot as plt

seed_all(36) # make reproducible

# create env | 创建环境
env = grid_world.GridWorldEnv(shape=(6, 10), initial_state=(0,0), target_state=(5,8))
agent = StateTableAgent("agent1", env=env)

# 初始化状态值 | initialize state values
initial_state_values = 100*torch.randn(len(env.observation_space))

# 学习 | learn
exit_code, (policy_table, Q_table, state_values), info = agent.learn(
    initial_state_values, gamma=0.9, eps_exit=1e-6,
    max_iter= 100, verbose_freq=10
)

# 绘制状态值 | plot state values
fig = grid_world.plot_state_values(env, state_values)
os.makedirs('./images', exist_ok=True)
fig.savefig('images/grid_world_state_values.png')

# 生成一个episode | generate an episode
episode, done = env.generate_episode(agent, (0,0))
print(f'{episode=}, {done=}')

# 绘制episode | plot episode
ani = grid_world.animate_episode(env, episode, interval=200)
file_to_save='images/grid_world_policy.gif' # None: do not save to file
ani.save(file_to_save, writer='pillow', fps=20)

plt.show()
```

Output:
```
 2024-09-04 16:26:06 | INFO | 10/100: dif-norm: 83.509293
 2024-09-04 16:26:06 | INFO | 20/100: dif-norm: 28.261782
 2024-09-04 16:26:06 | INFO | 30/100: dif-norm: 9.838606
 2024-09-04 16:26:06 | INFO | 40/100: dif-norm: 3.430509
 2024-09-04 16:26:06 | INFO | 50/100: dif-norm: 1.196144
 2024-09-04 16:26:06 | INFO | 60/100: dif-norm: 0.417070
 2024-09-04 16:26:06 | INFO | 70/100: dif-norm: 0.145423
 2024-09-04 16:26:06 | INFO | 80/100: dif-norm: 0.050706
 2024-09-04 16:26:06 | INFO | 90/100: dif-norm: 0.017679
 2024-09-04 16:26:06 | INFO | 100/100: dif-norm: 0.006164
 2024-09-04 16:26:06 | WARNING | Exit: Reach Max-Iter
episode=[((0, 0), '^', -1), ((1, 0), '^', -1), ((2, 0), '^', -1), ((3, 0), '^', -1), ((4, 0), '^', -1), ((5, 0), '->', -1), ((5, 1), '->', -1), ((5, 2), '->', -1), ((5, 3), '->', -1), ((5, 4), '->', -1), ((5, 5), '->', -1), ((5, 6), '->', -1), ((5, 7), '->', 0)], done=True
```

## Others
### Config
Config is a easy-to-use config loader with following features:
- light-weight and easy-to-use | 轻量级且简单易用
- missing key warning | 缺失key提醒 
- get value by nested key(s) .e.g. `method.optim` | 通过嵌套key(s)获取值，如`method.optim`
- check type and value range | 检查类型和值范围
- support check by field name | 支持通过字段名检查
- load from dict/json/yaml file | 从dict/json/yaml文件中加载
- save to json/yaml file | 保存为json/yaml文件

Simple: 
```python
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

```
More: [examples/demo_config_file.py](examples/demo_config_file.py)

## ChangeLog
- [@2024-08-23] 项目创建project created
- [@2024-08-28] add: state-value-iteration method
- [@2024-09-04] tag: v0.0.2: re-design: Agent/Env, StateTableAgent, PolicyTableAgent
- [@2024-09-05] tag: v0.0.3: QTableAgent, SarsaTableAgent
- [@2024-09-08] add: rlearn.utils.config: easy-to-use config loader
