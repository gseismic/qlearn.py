import numpy as np

# 环境设置
size = 5  # 网格世界的大小
n_actions = 4  # 四个动作：上、下、左、右
gamma = 0.9  # 折扣因子
alpha = 0.1  # 学习率
epsilon = 0.1  # ε-贪婪策略中的ε

# 初始化
Q = np.zeros((size, size, n_actions))  # 动作值函数 Q(s, a)
policy = np.ones((size, size, n_actions)) / n_actions  # 初始策略
returns = np.zeros((size, size, n_actions))  # 用于跟踪每个状态-动作对的回报

def generate_episode(policy):
    """根据策略生成一集"""
    episode = []
    state = (np.random.randint(size), np.random.randint(size))  # 随机起始状态
    while True:
        action_probs = policy[state[0], state[1]]
        action = np.random.choice(n_actions, p=action_probs)
        next_state = transition(state, action)
        reward = get_reward(next_state)
        episode.append((state, action, reward))
        if is_terminal(next_state):
            break
        state = next_state
    return episode

def transition(state, action):
    """根据动作计算下一个状态"""
    x, y = state
    if action == 0:  # 上
        return (max(x - 1, 0), y)
    elif action == 1:  # 下
        return (min(x + 1, size - 1), y)
    elif action == 2:  # 左
        return (x, max(y - 1, 0))
    elif action == 3:  # 右
        return (x, min(y + 1, size - 1))

def get_reward(state):
    """计算奖励，这里假设终点 (size-1, size-1) 的奖励为 1"""
    return 1 if state == (size - 1, size - 1) else 0

def is_terminal(state):
    """判断状态是否为终点"""
    return state == (size - 1, size - 1)

def update_policy(Q):
    """更新策略"""
    policy = np.zeros_like(Q)
    best_actions = np.argmax(Q, axis=2)
    for i in range(size):
        for j in range(size):
            best_action = best_actions[i, j]
            policy[i, j] = np.zeros(n_actions)
            policy[i, j, best_action] = 1.0
    return policy

# 蒙特卡洛方法进行策略评估和更新
def monte_carlo_policy_evaluation_and_improvement(episodes):
    global policy, Q
    for episode in episodes:
        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                Q[state[0], state[1], action] = Q[state[0], state[1], action] + alpha * (G - Q[state[0], state[1], action])
    policy = update_policy(Q)

# 运行蒙特卡洛方法
num_episodes = 3
for i in range(num_episodes):
    print(f'{i+1}/{num_episodes} ..')
    episode = generate_episode(policy)
    monte_carlo_policy_evaluation_and_improvement([episode])

print("优化后的策略：")
print(policy)
