import config
import torch
import matplotlib.pyplot as plt
from rlearn.envs import grid_world
from rlearn.utils.seed import seed_all
from rlearn.methods.table import QTableAgent

seed_all(36)

def test_qtable_agent_basic():
    env = grid_world.GridWorldEnv(shape=(6, 10), initial_state=(0,0), target_state=(5,8))
    agent = QTableAgent("agent1", env=env)

    initial_Q_table = torch.zeros(len(env.observation_space), len(env.action_space))
    learning_rate = 0.1
    n_epochs = 2000
    gamma = 0.9
    eps_explore = 0.1
    Q_eps_exit = 1e-6
    policy_eps_exit = 1e-6  
    check_exit_freq = 10
    max_timesteps_each_episode = None
    control_callback = None
    
    exit_code, (Q_table, policy_table, state_values), info = agent.learn(
        initial_Q_table=initial_Q_table, 
        learning_rate=learning_rate, 
        max_epochs=n_epochs, 
        gamma=gamma, 
        eps_explore=eps_explore, 
        Q_eps_exit=Q_eps_exit,
        policy_eps_exit=policy_eps_exit,
        check_exit_freq=check_exit_freq,
        max_timesteps_each_episode=max_timesteps_each_episode, 
        control_callback=control_callback
    )
    
    print(f'exit_code: {exit_code}')
    print(f'Q_table: {Q_table}')
    print(f'policy_table: {policy_table}')
    print(f'state_values: {state_values}')
    print(f'info: {info}')
    
    episode, done = env.generate_episode(agent, (0,0))
    print(f'episode: {episode}')
    print(f'done: {done}')
    
    ani = grid_world.animate_episode(env, episode, interval=200)
    plt.show()
    
    
if __name__ == "__main__":
    test_qtable_agent_basic()