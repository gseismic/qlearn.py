import config
import torch
from rlearn.envs import grid_world
from rlearn.utils.seed import seed_all
from rlearn.methods.table import PolicyTableAgent

seed_all(36)

def test_policy_agent_basic():
    env = grid_world.GridWorldEnv(shape=(6, 10), initial_state=(0,0), target_state=(5,8))
    agent = PolicyTableAgent("agent1", env=env)

    initial_policy_state_values = torch.zeros(len(env.observation_space))
    # initial_state_values = torch.randn(len(env.observation_space))
    initial_policy_table = torch.randn((len(env.observation_space),
                                  len(env.action_space)))
    # print(f'before: {initial_policy_table=}')
    initial_policy_table = torch.softmax(initial_policy_table, dim=1)
    # print(f'after: {initial_policy_table=}')
    gamma = 0.01 # 无法产生正确的结果
    gamma = 0.1 # 数值越小，越向目标space集中
    gamma = 0.5 # 数值越小，越向目标space集中，收敛越快
    verbose = 1
    exit_code, (policy_table, Q_table, state_values), info = agent.learn(
        initial_policy_table, gamma=gamma,
        eps_exit=1e-6,
        j_eps_exit=1e-3,
        j_trunc = 100,
        max_iter= 100, 
        verbose_freq=1
    )

    print(f'{state_values=}')
    # env.plot_state_values(state_values)

    def policy(state):
        i_state = env.index_of_state(state)
        max_index = torch.argmax(policy_table[i_state, :])
        action = env.action_space[max_index.item()]
        return action

    torch.set_printoptions(precision=9)
    print(f'{Q_table=}')
    print(f'{policy_table=}')
    episode, done = env.generate_episode(policy, (0,0))
    print(f'{episode=}')
    # 动画
    grid_world.animate_episode(env, episode, interval=500)

if __name__ == '__main__':
    if 1:
        test_policy_agent_basic()
