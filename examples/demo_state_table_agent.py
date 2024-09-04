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
ani = grid_world.animate_episode(env, episode, interval=500)
file_to_save='images/grid_world_policy.gif' # None: do not save to file
ani.save(file_to_save, writer='pillow', fps=1)

plt.show()
