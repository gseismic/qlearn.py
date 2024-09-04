import torch
from ...errcode import ExitCode
from ...logger import sys_logger


def value_iteration_learn(env, initial_state_values, gamma, eps_exit=1e-6, 
                          max_iter: int = 500, logger=None,
                          verbose_freq=10):
    """Learn by value iteration Method
    Args:
        eps_exit: exit if norm(V - prev_V) < eps_exit
        initial_state_values: initial state values
    """
    assert max_iter >= 1
    assert len(initial_state_values) == len(env.observation_space)
    states = env.observation_space
    actions = env.action_space
    logger = logger or sys_logger

    state_values = initial_state_values.clone()

    policy_table = torch.zeros(len(states), len(actions))
    Q_table = torch.zeros(len(states), len(actions))

    exit_code = -999
    i_iter = -1
    prev_state_values = torch.ones((len(states),))*float('-inf')
    while True:
        i_iter += 1
        if i_iter >= max_iter:
            exit_code = ExitCode.EXIT_REACH_MAX_ITER
            logger.warning(f'Exit: Reach Max-Iter: dif-norm: {chg_norm}')
            break

        value_dif = state_values - prev_state_values
        chg_norm = torch.norm(value_dif, p=2)
        if chg_norm < eps_exit:
            exit_code = ExitCode.EXIT_SUCC
            logger.info(f'Exit: Succ: norm: {chg_norm}')
            break
        elif (i_iter+1) % verbose_freq == 0:
            logger.info(f'{i_iter+1}/{max_iter}: dif-norm: {chg_norm:.6f}')

        for i, state in enumerate(states):
            for j, action in enumerate(actions):
                reward_ev = env.get_reward_ev(state, action)
                # 这样设计原因：reward可能是sparse的
                nextstate_value = env.get_nextstate_statevalue_ev(state, action,
                                                                  state_values)
                #for ii, nextstate in enumerate(states):
                #    state_p = env.get_state_prob(state, action, nextstate)
                #    nextstate_value += state_p * state_values[ii]
                Q_table[i, j] = reward_ev + nextstate_value * gamma

        max_values, max_indices = torch.max(Q_table, dim=1)
        # 更新策略
        policy_table[:, :] = 0
        policy_table[torch.arange(0, len(states)), max_indices] = 1
        prev_state_values = state_values.clone()
        state_values[:] = max_values # Q_table[:, max_indices]

    return exit_code, policy_table, (Q_table, state_values)
