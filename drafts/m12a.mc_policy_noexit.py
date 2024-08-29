import numpy as np
import random


class GridWorld:

    def __init__(self, size=4, terminal_states=[(0, 0), (3, 3)]):
        self.size = size
        self.terminal_states = terminal_states
        self.reset()

    def reset(self):
        """重置环境到起始状态"""
        self.state = (self.size - 1, 0)  # 开始在左下角
        return self.state

    def step(self, action):
        """根据给定的动作（上、下、左、右）更新状态，并返回奖励和下一个状态"""
        x, y = self.state
        if action == 0:  # 上
            next_state = (max(x - 1, 0), y)
        elif action == 1:  # 下
            next_state = (min(x + 1, self.size - 1), y)
        elif action == 2:  # 左
            next_state = (x, max(y - 1, 0))
        elif action == 3:  # 右
            next_state = (x, min(y + 1, self.size - 1))

        self.state = next_state
        if self.state in self.terminal_states:
            return next_state, 0, True  # 终止状态
        else:
            return next_state, -1, False  # 每步都给予一个负的奖励

def generate_episode(env, policy):
    """根据当前策略生成一个轨迹"""
    episode = []
    state = env.reset()
    while True:
        print(policy[state])
        action = np.random.choice(np.arange(len(policy[state])), p=policy[state])
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        print('\t\t', (state, action, reward))
        if done:
            break
    return episode

def mc_policy_iteration(env, episodes=1000, gamma=0.9):
    """蒙特卡洛策略迭代"""
    # 初始化
    Q = {}
    returns_sum = {}
    returns_count = {}
    policy = {}

    for i in range(env.size):
        for j in range(env.size):
            Q[(i, j)] = np.zeros(4)  # 4个动作
            policy[(i, j)] = np.ones(4) / 4  # 初始为均匀分布
            returns_sum[(i, j)] = np.zeros(4)
            returns_count[(i, j)] = np.zeros(4)

    # 开始迭代
    for episode_num in range(episodes):
        # 生成轨迹
        episode = generate_episode(env, policy)
        print(f'{episode=}')
        G = 0
        eps_greedy = 0.1
        for i in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[i]
            print(f'{i, state, action, reward =}')
            G = gamma * G + reward
            if not (state, action) in [(x[0], x[1]) for x in episode[:i]]:
                returns_sum[state][action] += G
                returns_count[state][action] += 1
                Q[state][action] = returns_sum[state][action] / returns_count[state][action]
                best_action = np.argmax(Q[state])
                policy[state] = np.eye(4)[best_action]

    return policy, Q

# 初始化环境
env = GridWorld()

# 执行蒙特卡洛策略迭代
policy, Q = mc_policy_iteration(env, episodes=10000, gamma=0.9)

# 输出最优策略和Q值
for i in range(env.size):
    for j in range(env.size):
        print(f"State ({i},{j}): Best action: {np.argmax(policy[(i, j)])}, Q-values: {Q[(i, j)]}")
