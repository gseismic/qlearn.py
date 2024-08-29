import torch
import numpy as np
import random

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

def epsilon_greedy_policy(state, q_table, epsilon, action_space):
    if random.random() < epsilon:
        return random.choice(action_space)
    else:
        q_values = q_table[state]
        return action_space[torch.argmax(q_values).item()]

def update_policy(q_table, epsilon, action_space):
    policy = {}
    for state in q_table:
        if np.random.rand() < epsilon:
            policy[state] = random.choice(action_space)
        else:
            best_action_index = torch.argmax(q_table[state]).item()
            policy[state] = action_space[best_action_index]
    return policy

def sarsa(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
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
            
            current_q = q_table[state][env.action_space().index(action)]
            next_q = q_table[next_state][env.action_space().index(next_action)]
            q_table[state][env.action_space().index(action)] = current_q + alpha * (reward + gamma * next_q - current_q)
            
            state = next_state
            action = next_action
        
        # 更新策略
        policy = update_policy(q_table, epsilon, env.action_space())
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} complete.")
            print("Current Policy:")
            for state, action in policy.items():
                print(f"State {state}: Action {action}")

    return q_table, policy

# 训练 SARSA 算法并更新策略
env = GridWorld(grid_size=4)
q_table, final_policy = sarsa(env, episodes=500)

# 打印最终 Q 表和策略
print("\nFinal Q Table:")
for state in env.state_space():
    print(f"State {state}: {q_table[state]}")

print("\nFinal Policy:")
for state, action in final_policy.items():
    print(f"State {state}: Action {action}")


def generate_episode_from_policy(env, policy, max_steps=100):
    """
    利用给定的策略生成一个 episode。

    参数:
    - env: 环境实例 (GridWorld)
    - policy: 最终策略 (字典，键为状态，值为动作)
    - max_steps: episode 最大步数

    返回:
    - episode: 包含 (状态, 动作, 奖励) 的列表
    """
    episode = []
    state = env.reset()  # 重置环境，获取初始状态

    for _ in range(max_steps):
        action = policy[state]  # 根据策略选择动作
        next_state, reward, done = env.step(action)  # 执行动作，获取下一个状态、奖励和是否结束
        
        episode.append((state, action, reward))  # 记录 (状态, 动作, 奖励)

        if done:
            break  # 如果到达终止状态，结束 episode
        
        state = next_state  # 更新状态
    
    return episode

# 生成一个 episode
episode = generate_episode_from_policy(env, final_policy)

# 打印生成的 episode
print("Generated Episode using final_policy:")
for step in episode:
    state, action, reward = step
    print(f"State: {state}, Action: {action}, Reward: {reward}")
