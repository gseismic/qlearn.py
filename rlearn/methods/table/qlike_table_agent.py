import torch
from abc import ABC, abstractmethod
from ..errcode import ExitCode
from ...core.agent import TableAgent

"""
Q-Learning Method | Q表学习方法
Algorithm:
```
1. initialize Q table Q(s, a) to arbitrary values (usually zero)
2. set learning rate α, discount factor γ, and exploration probability ε
3. initialize the environment and get the initial state s
4. repeat until termination condition:
   a. with probability ε select a random action a, or with probability 1 - ε select the action a that maximizes the Q value
   b. execute action a, and observe the next state s' and reward r
   c. calculate the target value: target = r + γ * max_a' Q(s', a')
   d. update the Q value: Q(s, a) ← Q(s, a) + α * (target - Q(s, a))
   e. update the current state to the next state s ← s'
5. end, return Q table Q(s, a)
```
---
```
1. 初始化 Q 值表 Q(s, a) 为任意值（通常为零）
2. 设定学习率 α、折扣因子 γ、和探索概率 ε
3. 初始化环境，并获取初始状态 s

4. 重复直到终止条件：
   a. 以概率 ε 选择一个随机动作 a，或者以概率 1 - ε 选择使得 Q 值最大的动作 a
   b. 执行动作 a，并观察到下一个状态 s' 和奖励 r
   c. 计算目标值：target = r + γ * max_a' Q(s', a')
   d. 更新 Q 值：Q(s, a) ← Q(s, a) + α * (target - Q(s, a))
   e. 将当前状态更新为下一个状态 s ← s'

5. 结束，返回 Q 值表 Q(s, a)
```
"""

class QLikeTableAgent(TableAgent, ABC):
    """Q-Learning Method | Q表学习方法
    
    """

    def __init__(self, name, env, *args, **kwargs):
        super(QLikeTableAgent, self).__init__(name=name, env=env, *args, **kwargs)

    def learn(self,
              initial_Q_table, 
              learning_rate,
              max_epochs,
              gamma=0.9,
              eps_explore=0.1,
              Q_eps_exit=1e-6,
              policy_eps_exit=1e-6,
              check_exit_freq=20,
              max_timesteps_each_episode=None,
              control_callback=None):
        """
        Q-Learning Method | Q表学习方法
        
        Args:
            initial_Q_table: initial Q table | 初始Q表
            learning_rate: learning rate | 学习率
            max_epochs: number of episodes | 轨迹个数
            gamma: discount factor | 折扣因子
            eps_explore: exploration probability | 探索概率
            max_timesteps_each_episode: maximum number of timesteps per episode | 每个轨迹的最大步数
            control_callback: 控制回调函数
        """ 
        assert 0 < eps_explore < 1
        self.Q_table = initial_Q_table.clone()
        self.policy_table = self.eps_greedy_policy_table_from_Q(self.Q_table, eps_explore)

        p_explore = 1/len(self.env.action_space) * eps_explore
        p_max = 1 - (len(self.env.action_space) - 1) * p_explore
        # print(f'{p_explore=}, {p_max=}, {p_explore * (n_action-1) + p_max=}')

        exit_code = ExitCode.EXIT_REACH_MAX_ITER
        prev_Q_table = torch.ones_like(self.Q_table) * float('-inf')
        prev_policy_table = torch.ones_like(self.policy_table) * float('-inf')
        for i in range(max_epochs):
            t = -1
            state, _ = self.env.reset()
            while True:
                t += 1
                i_state = self.env.index_of_state(state)
                i_action = torch.multinomial(self.policy_table[i_state, :], num_samples=1, replacement=True)[0]
                action = self.env.action_space[i_action]  # 以概率 ε 选择一个随机动作 a，或者以概率 1 - ε 选择使得 Q 值最大的动作 a       
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                i_next_state = self.env.index_of_state(next_state)
                i_next_action = torch.multinomial(self.policy_table[i_next_state, :], num_samples=1, replacement=True)[0]

                # update Q-table | 更新Q表
                self.update(
                    i_state, i_action, reward, i_next_state, i_next_action, learning_rate, gamma
                )

                # update policy | 更新策略
                max_index = torch.argmax(self.Q_table[i_state, :])
                self.policy_table[i_state, :] = p_explore
                self.policy_table[i_state, max_index] = p_max
            
                done = terminated is True or truncated is True
                if done:
                    break
                if max_timesteps_each_episode is not None and t >= max_timesteps_each_episode:
                    self.logger.warning(f'Episode {i} reached the maximum number of timesteps {max_timesteps_each_episode}')
                    break
                
                state = next_state
            
            if i % check_exit_freq == 0:
                Q_diff_norm = torch.norm(self.Q_table - prev_Q_table)
                policy_diff_norm = torch.norm(self.policy_table - prev_policy_table)
                self.logger.info(f'{i+1}/{max_epochs}: {Q_diff_norm=}, {policy_diff_norm=}')
                if Q_diff_norm < Q_eps_exit and policy_diff_norm < policy_eps_exit:
                    self.logger.info('Exit: Q table and policy table are stable')
                    exit_code = ExitCode.SUCC
                    break
                prev_Q_table = self.Q_table.clone()
                prev_policy_table = self.policy_table.clone()

        # Compute state values | 根据策略self.policy_table和Q表self.Q_table，计算状态值
        self.state_values = self.compute_state_values(self.Q_table, self.policy_table)
        info = {}
        return exit_code, (self.policy_table, self.Q_table, self.state_values), info

    @abstractmethod
    def update(self, i_state, i_action, reward, i_next_state, i_next_action, learning_rate, gamma):
        """
        更新 Q 表 | Update Q table

        Args:
            state: 当前状态 | Current state
            action: 当前动作 | Current action
            reward: 回报 | Reward
            next_state: 下一个状态 | Next state
            next_action: 下一个动作 | Next action
            learning_rate: 学习率 | Learning rate
            gamma: 折扣因子 | Discount factor
        """
        raise NotImplementedError()
