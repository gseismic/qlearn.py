

# Agent
class Policy:

    def __call__(self, state, *args, **kwargs):
        return self.forward(state, *args, **kwargs)

    def forward(self, state, *args, **kwargs):
        '''输出动作
        '''
        state_idx = self.state_indices[state]
        # action_idx = self.action_indices[action]
        p = self.policy[state_idx, action_idx]


import gymnasium as gym
env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
