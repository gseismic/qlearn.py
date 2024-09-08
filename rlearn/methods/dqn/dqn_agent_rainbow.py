from ...core.agent import Agent


class DQNAgent_Rainbow(Agent):

    def __init__(self, model,env, *args, **kwargs):
        super(DQNAgent_Rainbow, self).__init__(env, *args, **kwargs)
