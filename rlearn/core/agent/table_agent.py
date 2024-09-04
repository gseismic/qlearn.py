import torch
from .agent import Agent

class TableAgent(Agent):

    def __init__(self, name, env, *args, **kwargs):
        super(TableAgent, self).__init__(name=name, env=env, *args, **kwargs)
        states = self.env.observation_space
        actions = self.env.action_space

        self.state_values = torch.zeros((len(states),))
        self.Q_table = torch.zeros(len(states), len(actions))
        self.policy_table = torch.zeros(len(states), len(actions))

    def predict(self, state, deterministic=False):
        i_state = self.env.index_of_state(state)
        info = {}
        if deterministic:
            max_index = torch.argmax(self.policy_table[i_state, :])
            action = self.env.action_space[max_index.item()]
        else:
            i_action = torch.multinomial(self.policy_table[i_state, :], num_samples=1, replacement=True)[0]
            action = self.env.action_space[i_action]
        return action, info

    def __call__(self, state, deterministic=True):
        action, _ = self.predict(state, deterministic=deterministic)
        return action

    # @classmethod
    # def make_policy_table_from_Q(cls, Q_table, epsilon=0.1, greedy=False):
    #     """make policy table from Q table
    #     """
    #     n_actions = Q_table.shape[1]
    #     if greedy:
    #         p_explore = 0
    #         p_max = 1
    #     else:
    #         p_explore = 1/n_actions * epsilon
    #         p_max = 1 - (n_actions - 1) * p_explore
    #     max_values, max_indices = torch.max(Q_table, dim=1)
    #     policy_table = torch.ones_like(Q_table) * p_explore
    #     policy_table[torch.arange(0, len(states)), max_indices] = p_max
    #     return policy_table
    
    @classmethod
    def compute_state_values(cls, Q_table, policy_table):
        """make state values from Q table and policy table
        """
        state_values = torch.sum(policy_table * Q_table, dim=1)
        return state_values
    
    @classmethod
    def eps_greedy_policy_table_from_Q(cls, Q_table, eps=0.1):
        """make eps-greedy policy
        """
        n_states, n_actions = Q_table.shape
        p_explore = 1/n_actions * eps
        p_max = 1 - (n_actions - 1) * p_explore
        max_values, max_indices = torch.max(Q_table, dim=1)
        policy_table = torch.ones_like(Q_table) * p_explore
        policy_table[torch.arange(0, n_states), max_indices] = p_max
        return policy_table
