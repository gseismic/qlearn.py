import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ...core.env import TableEnv


class GridWorldEnv(TableEnv):

    def __init__(self, shape=(8, 5), initial_state=(0,0), target_state=None,
                 obstacles=None, *args, **kwargs):
        if isinstance(shape, (tuple, list)):
            self.shape = shape
        else:
            self.shape = (shape, shape)
        if obstacles is None:
            self.obstacles = set()
        elif isinstance(obstacles, (tuple, list)):
            self.obstacles = set(obstacles)
        else:
            raise TypeError(f'`type(obstacles)`(={type(obstacles)}) must (tuple,list)|None')
        self.target_state = target_state or (self.shape[0]-1, self.shape[1]-1)
        assert self.target_state not in self.obstacles

        action_space = ['^', 'v', '<-', '->']
        observation_space = [(i, j) for i in range(self.shape[0]) for j in range(self.shape[1])]
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         initial_state=initial_state, *args, **kwargs)
    
    def get_next_state(self, state, action):
        irow, icol = state
        if action == '^':
            next_state = (min(irow + 1, self.shape[0] - 1), icol)
        elif action == 'v':
            next_state = (max(irow - 1, 0), icol)
        elif action == '<-':
            next_state = (irow, max(icol - 1, 0))
        elif action == '->':
            next_state = (irow, min(icol + 1, self.shape[1] - 1))
        else:
            raise ValueError(f'未知动作(`{action}`)')

        if next_state in self.obstacles:
            next_state = copy.deepcopy(state)
        return next_state

    def get_reward(self, state, action, next_state):
        if state != next_state:
            if next_state == self.target_state:
                reward = 0
                reward_reason = 'target'
            else:
                reward = -1
                reward_reason = 'not-target'
        else:
            # encounter obstacles
            reward = -1
            reward_reason = 'obstacle'
        return reward, reward_reason
    
    def get_status(self, state, action, next_state, reward):
        status_code = 0
        if next_state == self.target_state:
            status_code = 1
        return status_code, {}

    def get_reward_ev(self, state, action):
        '''计算 sigma_{r} p(r|s, a) * r
        Args:
            state 当前状态
            action 当前状态采取的action
        '''
        next_state = self.get_next_state(state, action)
        reward, _ = self.get_reward(state, action, next_state)
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

    def to_dict(self):
        init = {
            'shape': self.shape,
            'initial_state': self.initial_state,
            'target_state': self.target_state,
            'obstacles': self.obstacles
        }
        env = {
            'name': self.name,
            'init': init,
            'state': self.state
        }
        return env

    @classmethod
    def from_dict(cls, dic):
        env = cls(dic['init'])
        if 'name' in env:
            env.set_name(dic['name'])
        if 'state' in env:
            env.set_state(dic['state'])
        return env
    
    def render(self):
        # TODO
        if self.render_mode == 'human':
            pass
        else:
            grid = np.zeros(self.shape)
            grid[self.state] = 1
            plt.imshow(grid, cmap='coolwarm', interpolation='none')
            plt.title('GridWorld')
            plt.show()
