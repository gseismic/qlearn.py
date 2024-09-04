import torch
from ...errcode import ExitCode
from ...logger import sys_logger

"""
离线版本的 SARSA（Offline SARSA）是指在离线设置中训练 SARSA 算法，其中所有的训练数据是预先收集的，不能在训练过程中进行在线更新。
离线 SARSA 通常用于在离线数据集上进行策略评估和优化。以下是离线版本的 SARSA 伪代码及其详细解释。

```
Initialize Q(s, a) arbitrarily for all states s and actions a
For each episode in the dataset:
    Initialize state s
    Choose action a from state s using policy derived from Q (e.g., epsilon-greedy)
    For each (state s, action a, reward r, next_state s', next_action a') in the episode:
        Compute the TD error: δ = r + γ * Q(s', a') - Q(s, a)
        Update Q-value: Q(s, a) = Q(s, a) + α * δ
        Set s = s', a = a'
```

伪代码要点
数据集：离线 SARSA 使用的是预先收集的数据集，其中每个 (state, action, reward, next_state, next_action) 对是固定的，不会在训练过程中动态更新。
Q 值更新：与在线 SARSA 相似，通过计算 TD 误差来更新 Q 值，但在离线 SARSA 中，所有的数据都是从固定的历史轨迹中获得的。
策略评估：由于数据是离线的，更新的 Q 值不会影响到实际的环境或策略选择，而只是基于预先收集的数据进行训练和评估。
"""

def sarsa_offline_learn(env, 
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
            i_next_action = I_state[t+1]
            q = Q_table[i_state, i_action]

            # sarsa
            q_next_sa = Q_table[i_next_state, i_next_action]
            q_hat = reward + gamma * q_next_sa

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

