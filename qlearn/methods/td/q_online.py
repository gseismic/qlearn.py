import torch
from ...errcode import ExitCode
from ...logger import sys_logger

"""
在线Q学习（Online Q-Learning）是一种强化学习算法，它通过与环境实时交互来更新Q值。与离线Q学习不同，在线Q学习在每次交互后立即更新Q值，而不是在收集完一批经验后进行更新。以下是在线Q学习的伪代码。


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

def Q_online_learn(env, 
                   initial_Q_table, 
                   initial_policy_table, 
                   alpha, 
                   num_episodes,
                   gamma,
                   eps_explore,
                   control_callback=None):
    """
    Args:
        num_episodes 轨迹个数
    """
    if control_callback is None:
        def fn_control():
            pass
        control_callback = fn_control

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

