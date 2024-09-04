import torch
from ...core.agent import Agent
from ...errcode import ExitCode


class TableAgent(Agent):

    def __init__(self, name, env, *args, **kwargs):
        super(TableAgent, self).__init__(name=name, env=env, *args, **kwargs)
        states = self.env.observation_space
        actions = self.env.action_space

        self.state_values = torch.zeros((len(states),))
        self.policy_table = torch.zeros(len(states), len(actions))
        self.Q_table = torch.zeros(len(states), len(actions))

    def predict(self, state, deterministic=False):
        i_state = self.env.index_of_state(state)
        info = {}
        if deterministic:
            max_index = torch.argmax(self.policy_table[i_state, :])
            action = self.env.action_space[max_index.item()]
        else:
            i_action = torch.multinomial(self.policy_table[i_state, :], num_samples=1, replacement=True)[0]
            action = self.env.action_space[i_action]
        return action, {}

    def __call__(self, state, deterministic=False):
        action, _ = self.predict(state, deterministic=deterministic)
        return action
