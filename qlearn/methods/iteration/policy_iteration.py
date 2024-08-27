import torch
from ..errcode import ExitCode
from ..logger import default_logger


def policy_iteration_learn(model, initial_policy, gamma, 
                           j_trunc=None,
                           eps_exit=1e-6, 
                           max_iter: int = 100, logger=None,
                           verbose_freq=10):
    """policy iteration Method

    Args:
        initial_policy: Pr, [n_state, n_action]
        eps_exit: exit if norm(V - prev_V) < eps_exit
    """
    assert max_iter >= 1
    assert len(initial_state_values) == len(model.states)
    assert initial_policy.shape == (len(model.states), len(model.actions))

    policy = initial_policy.clone()

    states = model.states
    actions = model.actions
    logger = logger or default_logger

    Q_table = torch.zeros(len(states), len(actions))

    prev_state_values = None
    i_iter = 0
    while True:
        if i_iter > max_iter:
            exit_code = ExitCode.EXIT_REACH_MAX_ITER
            logger.warning(f'Exit: Reach Max-Iter: norm: {chg_norm}')
            break

        i_iter += 1

        j = 0
        while True:
            if j_trunc is not None j >= j_trunc:
                break

            if prev_state_values is not None:
                value_dif = state_values - prev_state_values
                chg_norm = torch.norm(value_dif, p=2)
                if i_iter % verbose_freq == 0:
                    logger.info(f'{i_iter}/{max_iter}: norm: {chg_norm}')
                if chg_norm < eps_exit:
                    # exit_code = ExitCode.EXIT_SUCC
                    logger.info(f'Exit: Succ: norm: {chg_norm}')
                    break

            # Policy Evaluation: update `state_values`
            initial_state_values = torch.randn(len(states))
            for i, state in enumerate(states):
                state_values[i] = 0
                for k, action in enumerate(actions):
                    reward_ev = model.get_reward_ev(state, action)
                    nextstate_value = model.get_nextstate_value(state, action)
                    q = reward_ev + nextstate_value * gamma
                    state_values[i] += q * policy[i, k]

        # Policy Improvement
        for i, state in enumerate(states):
            for k, action in enumerate(actions):
                reward_ev = model.get_reward_ev(state, action)
                nextstate_value = model.get_nextstate_value(state, action)
                Q_table[i, k] = reward_ev + nextstate_value * gamma

            index_action_max = torch.argmax(Q_table, dim=1)
            # 更新策略
            policy[i, :] = 0
            policy[i, index_action_max] = 1

    return exit_code, policy, Q_table
