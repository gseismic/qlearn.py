import torch
from ..errcode import ExitCode
from ..logger import default_logger


def monto_carlo_explore(explore_eps):
    initial_q_values = None
    initial_q_table = None

    while True:
        initial_q_values


def mc_learn(model, initial_q_values, 
             initial_sa, gamma, 
             j_trunc=None, eps_explore=1e-6, 
             max_iter: int = 100, logger=None,
             verbose_freq=10):
    """Monto-Carlo (with eps-greedy option)

    Args:
        initial_q_values: Pr, [n_state, n_action]
        eps_explore: exit if norm(V - prev_V) < eps_explore
    """
    assert max_iter >= 1
    assert 0 < explore_eps <= 1
    assert len(initial_state_values) == len(model.states)
    assert initial_q_values.shape == (len(model.states), len(model.actions))
    logger = logger or default_logger

    model.policy = initial_q_values.clone()

    states = model.states
    actions = model.actions
    n_action = len(actions)

    p_explore = 1/n_action * eps_explore

    Q_table = torch.zeros(len(states), len(actions))
    Returns = torch.zeros(len(states), len(actions))
    Counts = torch.zeros(len(states), len(actions))
    G = torch.zeros(len(states), len(actions))
    while True:
        # 使用最新policy获得: (state, action, reward) 
        # 可统计每个(state,action)的访问次数
        # 相当于把reward，加权时间后，放回Q_table
        # 问题：
        #   如果对于某个状态s, 路径sar_list上 
        #       gen_policy 产生了动作次数为{a0:5, a1: 1, a2: 0}
        #       a1 次数不足，利用平均做估计值可能不准
        sar_list = [model.step() for i in range(T)]
        # (s0, a0, r1), (s1, a1, r2), ...
        I_states = torch.LongTensor([
            model.index_of_state(tu[0]) for tu in sar_list
        ])
        I_actions = torch.LongTensor([
            model.index_of_state(tu[1]) for tu in sar_list
        ])
        Rewards = torch.LongTensor([
            model.index_of_state(tu[2]) for tu in sar_list
        ])

        G.fill_(0)
        # G = gamma * G + 

        Counts[I_states, I_actions] += gamma * G
        for (state, action, reward) in reversed(sar_list):
            i_state = model.index_of_state(state)
            i_action = model.index_of_action(action)
            g = g * gamma + reward
            Return[i_state, i_action] += g
            # Q_table[

            # Policy Improvement
            # 获得每个状态最大Q值的动作，数值设置为1 - (n_action - 1) * p_explore
            # 其余设置一个比较小的探索概率
            # NOTE: 按照softmax with tempearature设置概率如何呢
            index_action_max = torch.argmax(Q_table, dim=1)
            model.policy[:, :] = 1 - (n_action - 1) * p_explore
            model.policy[:, index_action_max] = p_explore

