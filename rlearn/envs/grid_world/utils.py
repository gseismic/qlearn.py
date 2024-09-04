import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# NOTE FUTURE move to `renderer`
def animate_episode(env, episode, interval=200, fps=20,
                    show=False):
    """绘制策略执行的动画"""
    n_rows, n_cols = env.shape
    fig, ax = plt.subplots()
    grid = np.ones((n_rows, n_cols))
    agent_marker, = ax.plot([], [], 'ro', markersize=5)  # 红色圆圈表示智能体

    def init():
        """初始化函数，清空当前图像"""
        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        ax.imshow(grid, 
                  # cmap='coolwarm', 
                  cmap='Greys',
                  interpolation='none',
                  extent=[0, n_cols, 0, n_rows],
                  origin='lower') # 以0为中点
        return agent_marker,

    def update(frame):
        """更新函数，用于更新智能体的位置"""
        state = episode[frame][0]
        irow, icol = state
        agent_marker.set_data([icol+0.5], [irow+0.5])
        return agent_marker,

    # 修改 blit 为 False
    ani = animation.FuncAnimation(fig, update, frames=len(episode),
                                  init_func=init, blit=False, 
                                  repeat=False,
                                  interval=interval# 设置动画更新间隔，单位为毫秒
                                  )

    for i, (state, action, reward) in enumerate(episode):
        irow, icol = state
        ax.text(icol, irow, f'{action}:{reward}')

    #if file_to_save is not None:
    #    ani.save(file_to_save, writer='pillow', fps=fps)

    plt.title('GridWorld Policy-Trajectory')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')

    if show:
        plt.show()
    return ani

def plot_state_values(env, V, show=False):
    """绘制状态值函数的热图"""
    grid = np.zeros(env.shape)
    assert len(V) == len(env.observation_space)
    for k in range(len(V)):
        i, j = env.observation_space[k]
        grid[i, j] = V[k]

    fig = plt.figure()
    plt.imshow(grid, cmap='coolwarm', 
               extent=[0, env.shape[1], 0, env.shape[0]],
               interpolation='none',
               origin='lower') # 以0为中点

    plt.colorbar(label='State Value')
    plt.title('GridWorld State Value Function')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    if show:
        plt.show()
    return fig
