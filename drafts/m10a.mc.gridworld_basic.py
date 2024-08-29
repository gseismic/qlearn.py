import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('TkAgg')  # 使用 TkAgg 后端
from grid_world import GridWorld


def random_policy(state):
    """随机策略，返回一个随机动作"""
    return np.random.choice(4)


def generate_episode(env, policy):
    """生成一个完整的episode（从初始状态到终止状态）"""
    episode = []
    state = env.reset()

    while True:
        action = policy(state)
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break

    return episode


def monte_carlo_policy_evaluation(env, policy, num_episodes=500, gamma=1.0):
    """蒙特卡罗策略评估 state-value"""
    V = {}  # 值函数
    Returns = {}  # 回报集合

    # 初始化所有状态的返回集合
    for state in env.get_all_states():
        Returns[state] = []
        V[state] = 0

    # 生成episode并估计值函数
    for _ in range(num_episodes):
        episode = generate_episode(env, policy)
        G = 0  # 累积回报

        # 从episode终点开始回溯
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward  # 计算回报
            # 如果状态 s 在episode中首次出现
            if state not in [irow[0] for irow in episode[:t]]:
                Returns[state].append(G)
                V[state] = np.mean(Returns[state])  # 更新值函数

    return V


def plot_value_function(env, V, shape):
    """绘制状态值函数的热图"""
    grid = np.zeros(shape)

    for (i, j), value in V.items():
        # grid[j, i] = value  # 更新索引方式
        print(i,j)
        grid[i, j] = value  # 更新索引方式

    # plt.imshow(grid, cmap='coolwarm', interpolation='none', origin='lower')
    plt.imshow(grid, cmap='coolwarm', 
               extent=[0, env.shape[1], 0, env.shape[0]],
               interpolation='none',
               origin='lower') # 以0为中点

    plt.colorbar(label='State Value')
    plt.title('GridWorld State Value Function')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.show()


def plot_episode_animation(env, episode, interval=200):
    """绘制策略执行的动画"""
    fig, ax = plt.subplots()
    grid = np.zeros(env.shape)
    agent_marker, = ax.plot([], [], 'ro', markersize=5)  # 红色圆圈表示智能体

    def init():
        """初始化函数，清空当前图像"""
        ax.set_xlim([0, env.shape[1]])
        ax.set_xlim([0, env.shape[0]])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.grid(True)
        ax.imshow(grid, cmap='coolwarm', 
                  interpolation='none',
                  extent=[0, env.shape[1], 0, env.shape[0]],
                  origin='lower') # 以0为中点
        return agent_marker,

    def update(frame):
        """更新函数，用于更新智能体的位置"""
        state = episode[frame][0]
        irow, icol = state
        # print(irow, icol)
        agent_marker.set_data([irow+0.5], [icol+0.5])
        return agent_marker,

    # 修改 blit 为 False
    ani = animation.FuncAnimation(fig, update, frames=len(episode),
                                  init_func=init, blit=False, 
                                  repeat=False,
                                  interval=interval# 设置动画更新间隔，单位为毫秒
                                  )
    plt.show()


# 创建环境和策略
env = GridWorld(shape=(20, 30))
policy = random_policy

# 绘制策略运行的动画
if 0:
    episode = generate_episode(env, random_policy)
    print(episode)
    plot_episode_animation(env, episode, interval=200)

# 蒙特卡罗策略评估
V = monte_carlo_policy_evaluation(env, policy, num_episodes=5)

# 绘制状态值函数
plot_value_function(env, V, env.shape)
