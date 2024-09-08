from ...core.agent import Agent


class DQNAgent(Agent):

    def __init__(self, model,env, *args, **kwargs):
        super(DQNAgent, self).__init__(env, *args, **kwargs)
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.q_network = MLP(input_size=self.observation_space.shape[0],
                             output_size=self.action_space.n)
        self.target_network = MLP(input_size=self.observation_space.shape[0],
                                  output_size=self.action_space.n)
