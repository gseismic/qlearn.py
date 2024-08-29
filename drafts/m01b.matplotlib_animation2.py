import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 创建一个图形和坐标轴
fig, ax = plt.subplots()

size = 5
grid = np.zeros((size, size))
agent_marker, = ax.plot([], [], 'ro', markersize=10)  # 红色圆圈表示智能体

def init():
    """初始化函数，清空当前图像"""
    ax.set_xticks(np.arange(-0.5, size, 1))
    ax.set_yticks(np.arange(-0.5, size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #ax.set_xlim(0, 10)
    #ax.set_ylim(0, 10)
    ax.grid(True)
    ax.imshow(grid, cmap='coolwarm', origin='lower', alpha=0.3)
    return agent_marker,

def update(frame):
    """更新函数，用于更新智能体的位置"""
    state = episode[frame]
    x, y = state
    agent_marker.set_data([x], [y])
    print(state)
    return agent_marker,

episode = [(1, 2), (2, 2), (0, 1), (0, 0)]
# 修改 blit 为 False
ani = animation.FuncAnimation(fig, update, frames=len(episode),
                              init_func=init, blit=False, 
                              repeat=False,
                              interval=500  # 设置动画更新间隔，单位为毫秒
                              )


plt.show()
#plt.pause(3.1)  # 暂停以确保动画显示
print('111')
