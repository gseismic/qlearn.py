import torch
from ...errcode import ExitCode
from ...logger import sys_logger


def policy_iteration_learn(env, initial_policy_table, 
                           gamma, 
                           initial_policy_state_values=None,
                           j_trunc_min=2,
                           j_trunc=100,
                           j_eps_exit=1e-6, 
                           j_eval_method='gauss-seidel',
                           eps_exit=1e-6, 
                           exit_if_policy_stable=True,
                           max_iter: int = 500, 
                           logger=None,
                           j_verbose_freq=None,
                           verbose_freq=10,
                           verbose=1):
    """(vinila)policy iteration Method

    Note:
        - 本算法在state-value值变化小于特定数值后退出

    Args:
        - env 环境
        - initial_policy_table: [n_state, n_action] 初始策略
        - gamma 奖励折扣因子
        - j_trunc 截断数
        - j_trunc_min 截断前最低运行数
        - j_eps_exit 连续两次state-value的差值norm小于此值，退出内循环
        - eps_exit: 连续两次state-value的差值norm小于此值，退出【外循环】
        - exit_if_policy_stable 策略稳定即退出（可能导致state-value可能不太准确)
    """
    assert 0 < gamma <= 1.0
    assert j_eval_method in ['jacobi', 'gauss-seidel', 'gs']
    assert max_iter >= 1
    assert initial_policy_table.shape == (len(env.state_space),
                                          len(env.action_space))
    assert j_trunc is None or j_trunc > 0
    assert j_trunc_min is None or j_trunc_min >= 1
    if j_trunc_min is None and j_trunc is None:
        assert j_trunc_min <= j_trunc

    states, actions = env.observation_space, env.action_space
    logger = logger or sys_logger

    Q_table = torch.zeros(len(states), len(actions))
    if initial_policy_state_values is None:
        state_values = torch.randn(len(states))
    else:
        state_values = initial_policy_state_values.clone()

    # is_policy_stable = False
    prev_policy_table = initial_policy_table.clone()
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
            if j_eval_method in ['jacobi']:
                new_state_values = torch.zeros_like(state_values)
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
                    new_state_values[i] = new_i_state_value
                state_values = new_state_values.clone()
            elif j_eval_method in ['gauss-seidel', 'gs']:
                # gauss-seidel在单步迭代效果明显优于jacobi
                for i, state in enumerate(states):
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
                if j_trunc_min is None or j >= j_trunc_min:
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

        # 计算两次policy的state-value是否稳定, 如果稳定则退出
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

        if exit_if_policy_stable:
            # 计算两次policy的最大策略是否稳定, 如果稳定则退出
            # 策略稳定即退出，会导致Q_table 和 state-value精度下降
            max_values, max_indices = torch.max(policy_table, dim=1)
            prev_max_values, prev_max_indices = torch.max(prev_policy_table, dim=1)

            dif_max_action = max_indices - prev_max_indices
            dif_norm_policy = dif_max_action.float().norm(p=1)
            if dif_norm_policy.item() == 0:
                logger.info('Exit: policy stable')
                break
        #print(f'{max_indices=}')
        #print(f'{prev_max_indices=}')
        #print(f'{dif_norm_policy=}')
        #print(f'{dif_max_action=}')

        prev_policy_table = policy_table.clone()
        prev_policy_state_values = state_values.clone()

    return exit_code, policy_table, (Q_table, state_values)
