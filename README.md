# rlearn.py
Reinforcement Learning Library
强化学习库[在开发中 Developing]

## Reference
- gym 
- stable-baselines3
- more ..

## Methods
- [x] StateTableAgent | Value-Iteration Method
- [x] PolicyTableAgent | Policy-Iteration Method (with Truncate option)
- [ ] Sarsa
- [ ] Q-learning[on-policy]
- [ ] Q-learning[off-policy]
- [ ] DQN
- [ ] dual-DQN
- [ ] SAC
- [ ] DDPG
- [ ] HER
- [ ] Monte (REINFORCE)
- [ ] TRPO
- [ ] PPO
- [ ] A2C
- [ ] A3C
- [ ] QAC

## Examples 例子
Grid-World Demo
Use State-Value-Agent (State Value Iteration Method)
```
start: (0,0)
target: (5,8)
```
[state-values](examples/images/grid_world_state_values.png)
[policy](examples/images/grid_world_policy.gif)
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
## ChangeLog
- [@2024-08-23] 项目创建project created
- [@2024-08-28] add: state-value-iteration method
- [@2024-09-04] tag: v0.0.2: re-design: Agent/Env
