import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ...core.env import TableEnv, Env
from ...spaces import Box, Discrete


class GridWorldEnv_V2(Env):

    def __init__(self, 
                 shape=(5, 5), 
                 start_state=None, 
                 goal_state=None, 
                 obstacles=None,
                 reward_goal=1, 
                 reward_obstacle=-1, 
                 reward_step=-0.1,
                 *args, 
                 **kwargs):
        """
        Args:
            shape: tuple, grid size | 网格大小
            start_state: tuple, start state | 起始状态
            goal_state: tuple, goal state | 目标状态
            obstacles: tuple, obstacles | 障碍物
            reward_goal: float, reward for goal | 目标奖励
            reward_obstacle: float, reward for obstacle | 障碍物奖励
            reward_step: float, reward for step | 步长奖励
            render_mode: str, render mode (human|text) | 渲染模式
        """
        self.shape = shape
        self.start_state = start_state if start_state is not None else (0, 0)
        self.goal_state = goal_state if goal_state is not None else tuple(np.array(self.shape) - 1)
        if obstacles is None:
            obstacles = np.random.randint(low=0, high=np.array(self.shape), size=(3, 2))
            self.obstacles = set(map(tuple, obstacles.tolist()))
        elif isinstance(obstacles, (tuple, list)):
            self.obstacles = set(map(tuple, obstacles))
        else:
            raise TypeError(f'`type(obstacles)`(={type(obstacles)}) must (tuple,list)|None')
        assert goal_state not in self.obstacles, f'目标状态{self.goal_state}不能是障碍物'
        self.reward_goal = reward_goal
        self.reward_obstacle = reward_obstacle
        self.reward_step = reward_step
        
        # 0-up, 1-right, 2-down, 3-left
        self.action_space = Discrete(4)
        self.observation_space = Box(low=np.array([0, 0]), high=np.array(shape) - 1, dtype=np.int32)
        super(GridWorldEnv_V2, self).__init__(*args, **kwargs)

    def reset(self):
        """reset environment to initial state | 重置环境到起始状态
        Returns:
            state: current state (np.ndarray) | 当前状态
            info: extra info (dict) | 额外信息
        """
        self.previous_state = None
        self.current_state = np.array(self.start_state)
        return self.current_state, {}

    def step(self, action: int):
        """step environment | 根据动作更新状态，并返回下一个状态、奖励、是否完成和其他信息
        Args:
            action: action (int) | 动作
        Returns:
            next_state: next state (np.ndarray) | 下一个状态
            reward: reward (float) | 奖励
            terminated: bool, whether the episode is terminated | 是否终止
            truncated: bool, whether the episode is truncated | 是否截断
            info: dict, extra information | 额外信息
        """
        # down: (i, j) -> (i+1, j)
        # up: (i, j) -> (i-1, j)
        # left: (i, j) -> (i, j-1)
        # right: (i, j) -> (i, j+1)
        self.previous_state = self.current_state.copy()
        if action == 0:  # up   
            self.current_state[0] = max(0, self.current_state[0] - 1)
        elif action == 1:  # right
            self.current_state[1] = min(self.shape[1] - 1, self.current_state[1] + 1)
        elif action == 2:  # down
            self.current_state[0] = min(self.shape[0] - 1, self.current_state[0] + 1)
        elif action == 3:  # left
            self.current_state[1] = max(0, self.current_state[1] - 1)

        # 如果当前状态在障碍物中，则返回前一个状态
        terminated = False
        if tuple(self.current_state) in self.obstacles:
            self.current_state = self.previous_state.copy()
            reward = self.reward_obstacle
        elif np.array_equal(self.current_state, self.goal_state):
            reward = self.reward_goal
            terminated = True
        else:
            reward = self.reward_step
        
        truncated = False
        return self.current_state, reward, terminated, truncated, {}

    def render(self):
        """render environment | 渲染环境"""
        grid = np.full(self.shape, '.', dtype=str)
        grid[self.current_state[0], self.current_state[1]] = 'A'
        grid[self.goal_state[0], self.goal_state[1]] = 'G'
        
        for obstacle in self.obstacles:
            grid[obstacle[0], obstacle[1]] = 'X'
        
        for row in grid:
            print(' '.join(row))
        print()

    def close(self):
        """close environment | 关闭环境"""
        pass