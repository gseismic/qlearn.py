
class MCPGAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
    
    def learn(self, num_episodes, max_step_per_episode, max_total_steps, target_reward, seed=None):
        pass