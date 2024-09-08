import torch
from ...errcode import ExitCode
from ...logger import sys_logger

"""
离线Q学习（Offline Q-Learning）是一种经典的强化学习算法，用于通过与环境交互来学习一个最优策略。与在线Q学习不同，离线Q学习通常指在收集完一批经验后进行学习，而不是实时更新。以下是离线Q学习的伪代码。


```
1. 初始化 Q 值表 Q(s, a) 为任意值（通常为零）
2. 收集经验数据 D = {(s_i, a_i, r_i, s'_i) | i = 1, ..., N}，从环境中获取一批经验数据
3. 设定学习率 α、折扣因子 γ、和收敛阈值 θ

4. 重复直到收敛：
   a. 对于每条经验 (s, a, r, s') ∈ D：
      i. 计算目标值：target = r + γ * max_a' Q(s', a')
      ii. 更新 Q 值：Q(s, a) ← Q(s, a) + α * (target - Q(s, a))

5. 结束，返回 Q 值表 Q(s, a)
```
"""

def Q_offline_learn(env, 
                    initial_Q_table, 
                    initial_policy_table,
                    alpha, 
                    num_episodes,
                    gamma):
    """
    Args:
        num_episodes 轨迹个数
    """
    Q_table = initial_Q_table.clone()
    policy_table = initial_policy_table.clone()
    for i in range(num_episodes):

        policy = env.make_policy(policy_table)
        episode = env.generate_episode(policy)
        I_state = [env.index_of_state(sar[0]) for sar in episode]
        I_action = [env.index_of_state(sar[1]) for sar in episode]
        T = len(episode)

        # update Q-table
        for t in range(T-1):
            i_state = I_state[t]
            i_action = I_action[t]
            i_next_state = I_state[t+1]
            q = Q_table[i_state, i_action]
            max_q_next_state = torch.max(Q_table[i_next_state, :])
            q_hat = reward + gamma * max_q_next_state
            td_error = q - q_hat
            Q_table[i_state, i_action] = q - alpha * td_error

        # update policy
        max_values, max_indices = torch.max(Q_table, dim=1)
        # 更新策略
        policy_table[:, :] = 0
        policy_table[torch.arange(0, len(states)), max_indices] = 1

    state_values = None
    info = {}
    return exit_code, policy_table, (Q_table, state_values), info
