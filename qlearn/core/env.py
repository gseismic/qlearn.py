from abc import ABC
import numpy as np
import inspect


class Env:

    def __init__(self):
        self.reset()

    def reset(self):
        """重置环境，返回初始状态"""
        raise NotImplementedError()

    def step(self, action, agent=None):
        """`agent`执行了动作`action`后的状态，
        根据动作更新状态，返回新状态、奖励和终止标志
        Return:
            next_state
            reward
            done
        """
        raise NotImplementedError()


class EnvS(Env):

    def step(self, action):
        """`agent`执行了动作`action`后的状态，

        根据动作更新状态，返回新状态、奖励和终止标志
        Return:
            next_state
            reward
            done
        """
        raise NotImplementedError()

    def generate_episode(self, policy, reset=True, max_size=None):
        """生成一个完整的episode（从初始状态到终止状态）"""
        episode = []
        if reset:
            state = self.reset()

        #if not callable(inspect):
        #    fn_policy = lambda state: policy
        i = -1
        while True:
            i += 1
            action = policy(state)
            next_state, reward, done = self.step(action)
            # 不会记录target_state位置的
            episode.append((state, action, reward))
            state = next_state
            if done:
                episode.append((next_state, None, None))
                break
            elif max_size is not None and i >= max_size:
                episode.append((next_state, None, None))
                break

        return episode


class DiscreteEnv_S(EnvS):
    '''Discrete-Model for RL

    Model is Known:
        p(r|s, a)
        p(s'|s, a)
    '''

    def __init__(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space
        self.state_indices = {state: idx for idx, state in
                              enumerate(self._observation_space)}
        self.action_indices = {action: idx for idx, action in
                               enumerate(self._action_space)}
        super(DiscreteEnv_S, self).__init__()

    def index_of_state(self, state):
        return self.state_indices[state]

    def index_of_action(self, action):
        return self.action_indices[action]

    def state_of_index(self, index):
        return self._observation_space[index]

    def action_of_index(self, index):
        return self._action_space[index]

    def get_reward_ev(self, state, action):
        '''计算 sigma_{r} p(r|s, a) * r
        '''
        raise NotImplementedError()

    def get_nextstate_value(self, state, action):
        '''计算下一个状态的期望state-value
            sigma_{s'} p(s'|s, a) * value{s'}
        e.g.
                state_p = self.get_state_prob(state, action, next_state)
                next_state_value += state_p * state_values[ii]
        '''
        raise NotImplementedError()

    @property
    def random_policy(self):
        def policy(state):
            return np.random.choice(self.action_space)
        return policy

    @classmethod
    def _make_policy_from_table(cls, policy_table, eps_greedy=None):
        def policy(state):
            return np.random.choice(self.action_space)
        return policy

    @property
    def action_space(self):
        return self._action_space

    @property
    def state_space(self):
        return self._observation_space

    @property
    def observation_space(self):
        return self._observation_space


class EnvM:

    def step(self, action, agent=None):
        """`agent`执行了动作`action`后的状态，
        根据动作更新状态，返回新状态、奖励和终止标志
        Return:
            next_state
            reward
            done
        """
        raise NotImplementedError()
