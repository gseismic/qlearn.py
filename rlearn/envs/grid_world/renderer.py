

# TODO
class Renderer:

    def __init__(self, env):
        self.env = env

    def init(self):
        # TODO
        n_rows, n_cols = env.shape
        fig, ax = plt.subplots()
        grid = np.zeros((n_rows, n_cols))
        agent_marker, = ax.plot([], [], 'ro', markersize=5)  # 红色圆圈表示智能体
        pass

    def render(self):
        pass
