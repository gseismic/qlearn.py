import gym
import numpy as np
import matplotlib.pyplot as plt

# XXX 有错误
# 创建 Cliff Walk 环境
env = gym.make('CliffWalking-v0')

# 初始化Q表
Q = np.zeros([env.observation_space.n, env.action_space.n])

# 超参数
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索率
num_episodes = 500

# Q-Learning算法
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # 确保 state 是整数索引
        if isinstance(state, tuple):
            state = state[0] * env.observation_space.n // 4 + state[1]  # 映射到线性索引

        # ε-贪婪策略选择动作
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # 探索
        else:
            action = np.argmax(Q[state])  # 利用

        next_state, reward, done, _ = env.step(action)

        # 确保 next_state 是整数索引
        if isinstance(next_state, tuple):
            next_state = next_state[0] * env.observation_space.n // 4 + next_state[1]  # 映射到线性索引

        # Q值更新
        best_next_action = np.argmax(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])
        
        state = next_state

# 打印最终的Q表
print("Optimal Q-Table:")
print(Q)

# 绘制Q表的热图
plt.imshow(Q, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Heatmap of Q-Values")
plt.show()

