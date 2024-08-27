import torch
from ...errcode import ExitCode
from ...logger import sys_logger


def policy_iteration_learn(env, initial_policy_table, 
                           gamma, 
                           initial_policy_state_values=None,
                           j_trunc=100,
                           j_eps_exit=1e-6, 
                           eps_exit=1e-6, 
                           max_iter: int = 500, logger=None,
                           j_verbose_freq=None,
                           verbose_freq=10,
                           verbose=1):
    """(vinila)policy iteration Method

    Note:
        - 本算法在state-value值变化小于特定数值后退出

    Args:
        initial_policy_table: Pr, [n_state, n_action]
        eps_exit: exit if norm(V - prev_V) < eps_exit
    """
    assert max_iter >= 1
    assert initial_policy_table.shape == (len(env.state_space),
                                    len(env.action_space))
    assert j_trunc is None or j_trunc > 0

    states = env.observation_space
    actions = env.action_space
    logger = logger or sys_logger

    Q_table = torch.zeros(len(states), len(actions))
    if initial_policy_state_values is None:
        state_values = torch.randn(len(states))
    else:
        state_values = initial_policy_state_values.clone()

    # is_policy_stable = False
    policy_table = initial_policy_table.clone()
    prev_policy_state_values = torch.ones((len(states),)) * float('-inf')
    exit_code = -999
    i_iter = -1
    while True:
        # 每次迭代, 对于任意state, 均有v_policy_{k+1} >= v_policy_{k}
        #   state-value[折扣期望收益] 均有改进或保持不变
        i_iter += 1
        # 迭代法估算当前策略的state-value
        j = -1
        inner_prev_state_values = state_values.clone()
        # j_trunc 能否是chg_norm的函数
        j_chg_norm = None
        while True:
            j += 1
            # jacobi法更新 state_values
            for i, state in enumerate(states):
                # state_values[i] = 0
                new_i_state_value = 0
                for k, action in enumerate(actions):
                    reward_ev = env.get_reward_ev(state, action)
                    nextstate_value = env.get_nextstate_statevalue_ev(
                        state, action, state_values
                    )
                    # 该(state,action)下的：action-value
                    q = reward_ev + nextstate_value * gamma 
                    new_i_state_value += q * policy_table[i, k]
                state_values[i] = new_i_state_value

            value_dif = state_values - inner_prev_state_values
            j_chg_norm = torch.norm(value_dif, p=2)
            if verbose >= 2:
                if j_verbose_freq is not None and i_iter % verbose_freq == 0:
                    logger.debug(f'J-loop : {i_iter}/{max_iter}-j-{j+1}/{j_trunc}: norm: {j_chg_norm}')

            if j_chg_norm < j_eps_exit:
                if verbose >= 2:
                    logger.info(f'J-loop: Succ {j=}')
                break
            elif j_trunc is not None and j >= j_trunc: 
                if verbose >= 2:
                    logger.info(f'J-loop: Reach Max-Iter {j=}')
                break

            inner_prev_state_values = state_values.clone()

        # 更新 Q-table
        for i, state in enumerate(states):
            for k, action in enumerate(actions):
                reward_ev = env.get_reward_ev(state, action)
                nextstate_value = env.get_nextstate_statevalue_ev(
                    state, action, state_values
                )
                Q_table[i, k] = reward_ev + nextstate_value * gamma

        max_values, max_indices = torch.max(Q_table, dim=1)
        # 策略更新 Policy Improvement
        policy_table[:, :] = 0
        policy_table[torch.arange(0, len(states)), max_indices] = 1
        # state_values[:] = max_values # Q_table[:, max_indices]

        # 计算两次policy的state-value是否稳定
        value_dif = state_values - prev_policy_state_values
        chg_norm = torch.norm(value_dif, p=2)
        if verbose >= 1:
            if (i_iter+1) % verbose_freq == 0:
                logger.info(f'{i_iter+1}/{max_iter}: {j=}: dif-norm: {chg_norm:.6f}')

        if chg_norm < eps_exit:
            exit_code = ExitCode.EXIT_SUCC
            if verbose >= 1:
                logger.info('Succ')
            break
        elif i_iter >= max_iter:
            exit_code = ExitCode.EXIT_REACH_MAX_ITER
            if verbose >= 0:
                logger.warning('Exit: Reach Max-Iter')
            break

        prev_policy_state_values = state_values.clone()

    return exit_code, policy_table, (Q_table, state_values)

