import config
import torch
from rlearn.envs import grid_world
from rlearn.utils.seed import seed_all
from rlearn.methods.table import StateTableAgent

seed_all(36)


def test_state_agent_basic():
    env = grid_world.GridWorldEnv(shape=(6, 10), initial_state=(0,0), target_state=(5,8))
    agent = StateTableAgent("agent1", env=env)

    initial_state_values = torch.zeros(len(env.observation_space))
    # initial_state_values = torch.randn(len(env.observation_space))
    gamma = 0.9
    exit_code, (policy_table, Q_table, state_values), info = agent.learn(
        initial_state_values, gamma, eps_exit=1e-6,
        max_iter= 100, verbose_freq=1
    )

    print(state_values)
    grid_world.plot_state_values(env, state_values)

    def policy(state):
        i_state = env.index_of_state(state)
        max_index = torch.argmax(policy_table[i_state, :])
        action = env.action_space[max_index.item()]
        return action

    print(f'{Q_table=}')
    print(f'{policy_table=}')
    # policy = env.make_policy
    # episode, done = env.generate_episode(policy, (0,0))
    episode, done = env.generate_episode(agent, (0,0))
    print(f'{episode=}')
    # 动画
    grid_world.animate_episode(env, episode, interval=200)

if __name__ == '__main__':
    if 1:
        test_state_agent_basic()
