import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 创建一个图形和坐标轴
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', markersize=10)  # 红色圆点

# 初始化函数
def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return ln,

# 更新函数
def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

# 创建动画
ani = animation.FuncAnimation(
    fig,
    update,
    frames=np.arange(0, 10, 0.1),  # 动画帧的范围
    init_func=init,
    blit=True,  # 使用 blit 以优化动画绘制
    interval=100,  # 帧的时间间隔（毫秒）
)

plt.show()

