import torch
from ...core.agent import Agent
from ...errcode import ExitCode
from .table_agent import TableAgent


class StateTableAgent(TableAgent):

    def __init__(self, name, env, *args, **kwargs):
        super(StateTableAgent, self).__init__(name=name, env=env, *args, **kwargs)

    def learn(self, initial_state_values, gamma, 
              eps_exit=1e-6, 
              max_iter: int = 500, 
              verbose_freq=10):
        """Learn by value iteration Method

        Args:
            eps_exit: exit if norm(V - prev_V) < eps_exit
            initial_state_values: initial state values
        """
        assert max_iter >= 1
        assert len(initial_state_values) == len(self.env.observation_space)
        states = self.env.observation_space
        actions = self.env.action_space

        self.state_values = initial_state_values.clone()
        self.policy_table.fill_(0.0)
        self.Q_table.fill_(0.0)

        exit_code = ExitCode.UNDEFINED
        i_iter = -1
        prev_state_values = torch.ones((len(states),))*float('-inf')
        while True:
            i_iter += 1
            if i_iter >= max_iter:
                exit_code = ExitCode.EXIT_REACH_MAX_ITER
                self.logger.warning('Exit: Reach Max-Iter')
                break

            value_dif = self.state_values - prev_state_values
            chg_norm = torch.norm(value_dif, p=2)
            if chg_norm < eps_exit:
                exit_code = ExitCode.EXIT_SUCC
                self.logger.info(f'Exit: Succ: norm: {chg_norm}')
                break
            elif (i_iter+1) % verbose_freq == 0:
                self.logger.info(f'{i_iter+1}/{max_iter}: dif-norm: {chg_norm:.6f}')

            for i, state in enumerate(states):
                for j, action in enumerate(actions):
                    reward_ev = self.env.get_reward_ev(state, action)
                    # 这样设计原因：reward可能是sparse的
                    nextstate_value = self.env.get_nextstate_statevalue_ev(state, action,
                                                                      self.state_values)
                    #for ii, nextstate in enumerate(states):
                    #    state_p = env.get_state_prob(state, action, nextstate)
                    #    nextstate_value += state_p * state_values[ii]
                    self.Q_table[i, j] = reward_ev + nextstate_value * gamma

            max_values, max_indices = torch.max(self.Q_table, dim=1)
            # 更新策略
            self.policy_table[:, :] = 0
            self.policy_table[torch.arange(0, len(states)), max_indices] = 1
            prev_state_values = self.state_values.clone()
            self.state_values[:] = max_values # Q_table[:, max_indices]

        # return exit_code, policy_table, (Q_table, state_values)
        info = {}
        return exit_code, (self.policy_table, self.Q_table, self.state_values), info 
