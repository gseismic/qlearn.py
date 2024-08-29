import numpy as np
import random
from threading import Thread, Lock

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.state = (0, 0)
        self.goal = (size - 1, size - 1)
        self.actions = ['U', 'D', 'L', 'R']
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 'U':
            x = max(0, x - 1)
        elif action == 'D':
            x = min(self.size - 1, x + 1)
        elif action == 'L':
            y = max(0, y - 1)
        elif action == 'R':
            y = min(self.size - 1, y + 1)
        self.state = (x, y)
        reward = 1 if self.state == self.goal else 0
        done = self.state == self.goal
        return self.state, reward, done

class AsyncQLearning:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.q_table = {}
        self.lock = Lock()  # 用于同步更新的锁

        for state in [(i, j) for i in range(env.size) for j in range(env.size)]:
            self.q_table[state] = {action: 0.0 for action in env.actions}

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions)  # 探索
        else:
            return max(self.q_table[state], key=self.q_table[state].get)  # 利用

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        with self.lock:
            self.q_table[state][action] += self.alpha * td_error

    def train_agent(self, episodes):
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

def train_async_q_learning(env, num_agents, episodes_per_agent):
    agents = [AsyncQLearning(env) for _ in range(num_agents)]
    threads = []

    for agent in agents:
        thread = Thread(target=agent.train_agent, args=(episodes_per_agent,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # 合并所有代理的Q表
    q_table = {}
    for agent in agents:
        for state, actions in agent.q_table.items():
            if state not in q_table:
                q_table[state] = actions
            else:
                for action, value in actions.items():
                    q_table[state][action] = max(q_table[state][action], value)
    
    return q_table

# 创建环境和训练代理
env = GridWorld()
q_table = train_async_q_learning(env, num_agents=4, episodes_per_agent=1000)

# 打印训练后的Q值表
print("Trained Q-table:")
for state in sorted(q_table.keys()):
    print(f"State: {state}, Q-values: {q_table[state]}")


def extract_policy(q_table):
    policy = {}
    for state, actions in q_table.items():
        best_action = max(actions, key=actions.get)
        policy[state] = best_action
    return policy


# 提取和打印策略
policy = extract_policy(q_table)
print("Extracted Policy:")
for state in sorted(policy.keys()):
    print(f"State: {state}, Best Action: {policy[state]}")


def generate_trajectory(env, policy, start_state):
    trajectory = []
    state = start_state
    done = False
    
    while not done:
        trajectory.append(state)
        action = policy.get(state)
        if action is None:
            break
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
    
    trajectory.append(state)  # Append the final state
    return trajectory

# 生成轨迹
start_state = (0, 0)
trajectory = generate_trajectory(env, policy, start_state)

# 打印轨迹
print("\nGenerated Trajectory:")
for step in trajectory:
    print(step)
