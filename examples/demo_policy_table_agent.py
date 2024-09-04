import torch
from rlearn.envs import grid_world
from rlearn.utils.seed import seed_all
from rlearn.methods.table import PolicyTableAgent
import matplotlib.pyplot as plt
seed_all(36)
torch.set_printoptions(precision=9)

# 创建环境 | create env
env = grid_world.GridWorldEnv(shape=(6, 10), initial_state=(0,0), target_state=(5,8))
agent = PolicyTableAgent("agent1", env=env)

# 初始化策略表 | initialize policy table
initial_policy_table = torch.randn((len(env.observation_space), len(env.action_space)))
initial_policy_table = torch.softmax(initial_policy_table, dim=1)

# 学习 | learn
exit_code, (policy_table, Q_table, state_values), info = agent.learn(
    initial_policy_table, 
    gamma=0.5, # 数值越小，越向目标space集中，收敛越快
    eps_exit=1e-6,
    j_eps_exit=1e-3,
    j_trunc = 100,
    max_iter= 100, 
    verbose_freq=1
)

# 生成一个episode | generate an episode
episode, done = env.generate_episode(agent, (0,0))
print(f'{episode=}')

fig = grid_world.plot_state_values(env, state_values)
#fig.savefig('grid_world_state_values.png')

# file_to_save='grid_world_policy2.gif' # None: do not save to file
ani = grid_world.animate_episode(env, episode, interval=200)
# ani.save(file_to_save, writer='pillow', fps=20)

plt.show()
