import gymnasium as gym
import numpy as np
import pytest
from rlearn.utils.seed import seed_all
from rlearn.methods.mcpg.naive import MCPGAgent_Naive

def test_mcpg_naive_initialization():
    env = gym.make('CartPole-v1')
    config = {
        'learning_rate': 0.01,
        'gamma': 0.99,
        'normalize_return': True,
        'policy.hidden_sizes': [64, 64],
        'policy.activation': 'tanh'
    }
    agent = MCPGAgent_Naive(env, config=config)
    assert agent.config['learning_rate'] == 0.01
    assert agent.config['gamma'] == 0.99
    assert agent.config['normalize_return'] == True
    assert agent.config['policy.hidden_sizes'] == [64, 64]
    assert agent.config['policy.activation'] == 'tanh'
    assert agent.env == env
    print(agent.config)

def test_mcpg_naive_learning():
    env = gym.make('CartPole-v1')
    seed_all(36)
    
    config = {
        'learning_rate': 0.01,
        'gamma': 0.99,
        'normalize_return': True,
        'hidden_sizes': [64, 64],
        'activation': 'tanh'
    }
    agent = MCPGAgent_Naive(env, config=config)
    
    initial_performance = evaluate_agent(agent, env)
    # agent.learn(num_episodes=100, seed=36)
    exit_info = agent.learn(num_episodes=300, 
                            max_step_per_episode=500, 
                            max_total_steps=150000, 
                            target_episode_reward=195)
    print(f'{exit_info=}')
    final_performance = evaluate_agent(agent, env)
    assert final_performance > initial_performance

def test_mcpg_naive_deterministic():
    seed_all(36)
    env = gym.make('CartPole-v1')    
    config = {
        'learning_rate': 0.01,
        'gamma': 0.99,
        'normalize_return': True,
        'hidden_sizes': [64, 64],
        'activation': 'tanh'
    }
    agent1 = MCPGAgent_Naive(env, config=config, seed=36)
    agent1.learn(num_episodes=100)
    final_performance1 = evaluate_agent(agent1, env)
    
    agent2 = MCPGAgent_Naive(env, config=config, seed=36)
    agent2.learn(num_episodes=100)
    final_performance2 = evaluate_agent(agent2, env)
    
    print(f'{final_performance1=}')
    print(f'{final_performance2=}')
    assert np.allclose(final_performance1, final_performance2)
    # assert np.allclose(agent1.policy.get_weights(), agent2.policy.get_weights())
    
def evaluate_agent(agent, env, n_episodes=10):
    total_rewards = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards += episode_reward
    return total_rewards / n_episodes

if __name__ == '__main__':
    # pytest.main([__file__])
    if 0:
        test_mcpg_naive_initialization()
    if 0:
        test_mcpg_naive_learning()
    if 1:
        test_mcpg_naive_deterministic()
    if 0:
        test_mcpg_naive_different_configs()