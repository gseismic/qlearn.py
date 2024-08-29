import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ...core.env import DiscreteEnv_S


class GridWorld(DiscreteEnv_S):

    def __init__(self, shape=(8, 5), 
                 initial_state=None, target_state=None,
                 fn_reward=None):
        if isinstance(shape, (tuple,list)):
            self.shape = shape
        else:
            self.shape = (shape, shape)
        self.initial_state = initial_state or (0, 0)
        self.target_state = target_state or (self.shape[0]-1, self.shape[1]-1)

        if fn_reward is None:
            self.fn_reward = lambda s, a, n_s: float(n_s == self.target_state)
        else:
            self.fn_reward = fn_reward
        action_space = ['^', 'v', '<-', '->']
        observation_space = [(i, j) for i in range(self.shape[0]) for j in range(self.shape[1])]
        super(GridWorld, self).__init__(observation_space=observation_space,
                                        action_space=action_space)

    def reset(self, state=None):
        self.state = self.initial_state if state is None else state
        return self.state

    def get_next_state_probs(self, state, action):
        # 以概率分布方式输出下一个状态分布
        next_state = self.get_next_state(state, action)
        return {next_state: 1.0}

    def get_next_state(self, state, action):
        # 更广义的应该是一个概率分布
        # print(f'get_next_state: {state, action=}')
        assert action in self.action_space
        irow, icol = state
        if action == '^': # 上
            next_state = (min(irow + 1, self.shape[0] - 1), icol)
        elif action == 'v':   # 下
            next_state = (max(irow - 1, 0), icol)
        elif action == '<-': # 左
            next_state = (irow, max(icol - 1, 0))
        elif action == '->': # 右
            next_state = (irow, min(icol + 1, self.shape[1]- 1))
        else:
            raise ValueError(f'unknown action(`{action}`)')

        return next_state

    def get_reward(self, state, action, next_state):
        reward = self.fn_reward(self.state, action, next_state)
        return reward

    def step(self, action):
        # print(f'step: {self.state, action=}')
        next_state = self.get_next_state(self.state, action)
        reward = self.fn_reward(self.state, action, next_state)
        if next_state == self.target_state:
            done = True
        else:
            done = False
        self.state = next_state
        return next_state, reward, done

    def get_reward_ev(self, state, action):
        '''计算 sigma_{r} p(r|s, a) * r
        Args:
            state 当前状态
            action 当前状态采取的action
        '''
        next_state = self.get_next_state(state, action)
        reward = self.get_reward(state, action, next_state)
        return reward

    def get_nextstate_statevalue_ev(self, state, action, state_values):
        '''计算下一个状态的期望state-value
            sigma_{s'} p(s'|s, a) * value{s'}
        e.g.
                state_p = self.get_state_prob(state, action, next_state)
                next_state_value += state_p * state_values[ii]
        '''
        # 因为本例只有一个状态
        next_state = self.get_next_state(state, action)
        i_next_state = self.index_of_state(next_state)
        statevalue_ev = state_values[i_next_state]
        return statevalue_ev
        #i_state = self.index_of_state(state)
        #i_action = self.index_of_action(action)

    def animate_episode(self, episode, interval=200):
        """绘制策略执行的动画"""
        n_rows, n_cols = self.shape
        fig, ax = plt.subplots()
        grid = np.zeros((n_rows, n_cols))
        agent_marker, = ax.plot([], [], 'ro', markersize=5)  # 红色圆圈表示智能体

        def init():
            """初始化函数，清空当前图像"""
            #ax.set_xlim([0, n_cols])
            #ax.set_ylim([0, n_rows])
            ax.set_xticks(range(n_cols))
            ax.set_yticks(range(n_rows))
            #ax.set_xticklabels([])
            #ax.set_yticklabels([])
            ax.grid(True)
            ax.imshow(grid, cmap='coolwarm', 
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

        plt.show()


    def plot_state_values(self, V):
        """绘制状态值函数的热图"""
        grid = np.zeros(self.shape)
        assert len(V) == len(self.observation_space)
        for k in range(len(V)):
            i, j = self.observation_space[k]
            grid[i, j] = V[k]

        plt.imshow(grid, cmap='coolwarm', 
                   extent=[0, self.shape[1], 0, self.shape[0]],
                   interpolation='none',
                   origin='lower') # 以0为中点

        plt.colorbar(label='State Value')
        plt.title('GridWorld State Value Function')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.show()
