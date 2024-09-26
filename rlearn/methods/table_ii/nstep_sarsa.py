import torch
import numpy as np
from rlearn.methods.table_ii.base_agent import BaseAgent

class NStepSARSAAgent(BaseAgent):
    """
    N-Step SARSA Agent
    """
    schema = [
        dict(field='n_step', required=False, default=3, rules=dict(type='int', gt=0)),
        dict(field='use_strict_n_step', required=False, default=False, rules=dict(type='bool')),
        dict(field='learning_rate', required=False, default=0.1, rules=dict(type='float', gt=0, max=1)),
        dict(field='gamma', required=False, default=0.99, rule=dict(type='float', min=0, max=1)),
        dict(field='epsilon', required=False, default=0.1, rules=dict(type='float', min=0, max=1)),
        dict(field='verbose_freq', required=False, default=10, rules=dict(type='int', gt=0)),
    ]
    def __init__(self, env, config=None, logger=None):
        super().__init__(env, config, logger)
        
    def select_action(self, state):
        if np.random.rand() < self.config['epsilon']:
            return np.random.choice(self.n_actions)
        else:
            return torch.argmax(self.Q_table[state]).item()

    def update(self, trajectory):
        if self.config['use_strict_n_step'] is True and len(trajectory) < self.config['n_step']:
            return False
        
        current_n_step = len(trajectory)
        assert 0 < current_n_step <= self.config['n_step']
        G = 0
        for i, (state, action, reward) in enumerate(trajectory):
            G += (self.gamma ** i) * reward
        
        state, action, _ = trajectory[0]
        next_state, next_action, _ = trajectory[-1]

        G += (self.gamma ** current_n_step) * self.Q_table[next_state][next_action].item()
        td_error = G - self.Q_table[state][action]
        self.Q_table[state][action] += self.config['learning_rate'] * td_error
        return True
