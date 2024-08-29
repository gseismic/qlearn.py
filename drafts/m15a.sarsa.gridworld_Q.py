import numpy as np

class GridWorld:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.state = (0, 0)
        self.end_state = (grid_size - 1, grid_size - 1)
        self.actions = ['up', 'down', 'left', 'right']
    
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

import torch
import random

def epsilon_greedy_policy(state, q_table, epsilon, action_space):
    if random.random() < epsilon:
        return random.choice(action_space)
    else:
        q_values = q_table[state]
        return action_space[torch.argmax(q_values).item()]

def sarsa(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    # 初始化 Q 表，状态-动作对的 Q 值初始化为 0
    q_table = {}
    for state in env.state_space():
        q_table[state] = torch.zeros(len(env.action_space()), dtype=torch.float32)
    
    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy_policy(state, q_table, epsilon, env.action_space())
        
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_policy(next_state, q_table, epsilon, env.action_space())
            
            # SARSA 更新公式
            current_q = q_table[state][env.actions.index(action)]
            next_q = q_table[next_state][env.actions.index(next_action)]
            q_table[state][env.actions.index(action)] = current_q + alpha * (reward + gamma * next_q - current_q)
            
            state = next_state
            action = next_action

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} complete.")

    return q_table

def update_policy(q_table, epsilon, action_space):
    """
    根据当前的 Q 表和 epsilon 值更新策略。

    参数：
    - q_table: 当前的 Q 值表，存储为字典 {state: Q_values}
    - epsilon: epsilon 值，用于 epsilon-greedy 策略
    - action_space: 动作空间列表

    返回：
    - policy: 更新后的策略，存储为字典 {state: action}
    """
    policy = {}

    for state in q_table:
        # 使用 epsilon-greedy 策略
        if np.random.rand() < epsilon:
            # 探索：随机选择一个动作
            policy[state] = random.choice(action_space)
        else:
            # 利用：选择 Q 值最大的动作
            best_action_index = torch.argmax(q_table[state]).item()
            policy[state] = action_space[best_action_index]

    return policy


# 训练 SARSA 算法
env = GridWorld(grid_size=4)
q_table = sarsa(env, episodes=500)

# 打印 Q 表
for state in env.state_space():
    print(f"State {state}: {q_table[state]}")

#==============================================================================================
# 初始化环境和参数
env = GridWorld(grid_size=4)
q_table = sarsa(env, episodes=500)
epsilon = 0.1
action_space = env.action_space()

# 更新策略
policy = update_policy(q_table, epsilon, action_space)

# 打印更新后的策略
for state, action in policy.items():
    print(f"State {state}: Action {action}")
