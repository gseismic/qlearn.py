import gymnasium as gym
from rlearn.utils.seed import seed_all
from rlearn.methods.table_ii.nstep_q import NStepQAgent

"""
objective: test N-step Q-learning
"""

seed_all(36)

def test_nstep_q():
    env = gym.make('CliffWalking-v0')
    config = {
        'n_step': 10,
        'use_strict_n_step': True,
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon': 0.1,
        'verbose_freq': 1
    }
    q_learning = NStepQAgent(env, config=config)
    seed = None
    num_episodes = 500
    q_learning.learn(num_episodes=num_episodes, max_step_per_episode=500, max_total_steps=100000, target_reward=500, seed=seed)
    # print("Final Q-Table:")
    # print(q_learning.Q_table)

if __name__ == '__main__':
    if 1:
        test_nstep_q()
