from .qlike_table_agent import QLikeTableAgent
"""
Algorithm【Sarsa】
Initialize Q(s, a) arbitrarily for all states s and actions a
For each episode:
    Initialize state s
    Choose action a from state s using policy derived from Q (e.g., epsilon-greedy)
    For each step of the episode:
        Take action a, observe reward r, and next state s'
        Choose next action a' from state s' using policy derived from Q
        Compute the TD error: δ = r + γ * Q(s', a') - Q(s, a)
        Update Q-value: Q(s, a) = Q(s, a) + α * δ
---
Algorithm【Sarsa】
初始化Q表
对于每个轨迹：
    初始化状态s
    使用Q表派生策略选择动作a
    对于每个轨迹步骤：
        采取动作a，观察奖励r和下一个状态s'
        使用Q表派生策略选择下一个动作a'
        计算TD误差：δ = r + γ * Q(s', a') - Q(s, a)
        更新Q值：Q(s, a) = Q(s, a) + α * δ
        设置s = s', a = a'
        如果s是终止状态：
"""

class SarsaTableAgent(QLikeTableAgent):
    """Sarsa Method | Sarsa表学习方法
    
    """

    def __init__(self, name, env, *args, **kwargs):
        super(SarsaTableAgent, self).__init__(name=name, env=env, *args, **kwargs)  
    
    def update(self, i_state, i_action, reward, i_next_state, i_next_action, learning_rate, gamma):
        """
        Sarsa 更新 Q 表 | Update Q table

        Args:
            state: 当前状态 | Current state
            action: 当前动作 | Current action
            reward: 回报 | Reward
            next_state: 下一个状态 | Next state
            next_action: 下一个动作 | Next action (可选 | Optional)
            learning_rate: 学习率 | Learning rate
            gamma: 折扣因子 | Discount factor
        """
        q = self.Q_table[i_state, i_action]
        q_next = self.Q_table[i_next_state, i_next_action]
        q_hat = reward + gamma * q_next
        td_error = q - q_hat
        self.Q_table[i_state, i_action] = q - learning_rate * td_error
