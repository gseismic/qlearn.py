import numpy as np
import random
from collections import deque

class GridWorld:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.state = (0, 0)
        self.end_state = (grid_size - 1, grid_size - 1)
        self.actions = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.actions)
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        x, y = self.state
        
        if action == 'up':
            x = max(x - 1, 0)
        elif action == 'down':
            x = min(x + 1, self.grid_size - 1)
        elif action == 'left':
            y = max(y - 1, 0)
        elif action == 'right':
            y = min(y + 1, self.grid_size - 1)
        
        self.state = (x, y)
        if self.state == self.end_state:
            return self.state, 1, True  # Reward of 1 when reaching the end
        else:
            return self.state, 0, False  # No reward otherwise

    def action_space(self):
        return self.actions

    def state_space(self):
        return [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]

def generate_n_step_episode(env, policy, n, gamma=0.9):
    episode = []
    state = env.reset()
    action = policy[state]
    states = [state]
    actions = [action]
    rewards = []
    dones = []
    
    for _ in range(n):
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward, next_state, done))
        state, action = next_state, policy[next_state]
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        if done:
            break

    return episode, states, actions, rewards, dones

def n_step_sarsa(env, policy, n, gamma=0.9, alpha=0.1, episodes=1000):
    Q = {state: {action: 0 for action in env.action_space()} for state in env.state_space()}
    
    for _ in range(episodes):
        episode, states, actions, rewards, dones = generate_n_step_episode(env, policy, n, gamma)
        T = len(episode)
        for t in range(T):
            G = sum([rewards[t+i] * (gamma ** i) for i in range(n) if t+i < T])  # n-step return
            if t + n < T:
                G += (gamma ** n) * Q[states[t+n]][actions[t+n]]
            state, action, _, _, done = episode[t]
            Q[state][action] += alpha * (G - Q[state][action])
        
        # Update policy based on Q
        for state in env.state_space():
            best_action = max(Q[state], key=Q[state].get)
            policy[state] = best_action
    
    return policy, Q

# 使用示例
env = GridWorld(grid_size=4)
n = 3  # n步的数量
gamma = 0.9
alpha = 0.1
episodes = 1000

# 初始化策略
policy = {state: random.choice(env.action_space()) for state in env.state_space()}

# 执行n步sarsa策略迭代
policy, Q = n_step_sarsa(env, policy, n, gamma, alpha, episodes)

# 打印最终策略和Q值
print("Optimal Policy:")
for state, action in policy.items():
    print(f"State: {state}, Action: {action}")

print("\nQ Function:")
for state in env.state_space():
    for action in env.action_space():
        print(f"State: {state}, Action: {action}, Q-value: {Q[state][action]:.2f}")

