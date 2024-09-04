import torch
from .qlike_table_agent import QLikeTableAgent

class QTableAgent(QLikeTableAgent):
    """Q-Learning Method | Q表学习方法
    
    """

    def __init__(self, name, env, *args, **kwargs):
        super(QTableAgent, self).__init__(name=name, env=env, *args, **kwargs)  
    
    def update(self, i_state, i_action, reward, i_next_state, i_next_action, learning_rate, gamma):
        """
        更新 Q 表 | Update Q table

        Args:
            state: 当前状态 | Current state
            action: 当前动作 | Current action
            reward: 回报 | Reward
            next_state: 下一个状态 | Next state
            next_action: 下一个动作 | Next action (可选 | Optional)
        """
        # update Q-table | 更新Q表
        q = self.Q_table[i_state, i_action]
        q_hat = reward + gamma * torch.max(self.Q_table[i_next_state, :]).item()
        td_error = q - q_hat
        self.Q_table[i_state, i_action] = q - learning_rate * td_error
        