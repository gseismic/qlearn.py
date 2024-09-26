import gymnasium as gym
from rlearn.utils.seed import seed_all
from rlearn.methods.table_ii.nstep_sarsa import NStepSARSAAgent

"""
objective: test N-step SARSA
"""

seed_all(36)

def test_nstep_sarsa():
    env = gym.make('CliffWalking-v0')
    config = {
        'n_step': 10,
        'use_strict_n_step': False,
        'learning_rate': 0.1,
        'gamma': 0.99,
        'epsilon': 0.1,
        'verbose_freq': 1
    }
    sarsa_agent = NStepSARSAAgent(env, config=config)
    seed = None
    num_episodes = 50
    sarsa_agent.learn(num_episodes=num_episodes, 
                      max_step_per_episode=500, 
                      max_total_steps=100000, 
                      target_episode_reward=500, 
                      seed=seed)
    # print("Final Q-Table:")
    # print(sarsa_agent.Q_table)

if __name__ == '__main__':
    test_nstep_sarsa()