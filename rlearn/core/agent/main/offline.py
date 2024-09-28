from rlearn.core.agent.main.base import BaseAgent

class OfflineAgent(BaseAgent):
    schema = []

    def __init__(self, env, config=None, logger=None, seed=None):
        super().__init__(env, config, logger, seed)
        self.init()
    
    def init(self):
        pass