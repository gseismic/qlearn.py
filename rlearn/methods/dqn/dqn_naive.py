from ...core.agent import Agent
from ...utils.replay_buffer import ReplayBuffer


class DQNAgent_OnPolicy_Naive(Agent):
    """
    DQN Agent Naive [online-version] | 朴素DQN [online版本]
    """
    def __init__(self, name, env, model, batch_size, *args, **kwargs):
        super(DQNAgent_OnPolicy_Naive, self).__init__(name=name, env=env, *args, **kwargs)
        self.model = model
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = batch_size

    def _choose_action(self, state, epsilon):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample() # 随机选择动作
        else:
            q_action = self.model(state) # shape: (1, n_actions)
            i_action = torch.argmax(q_action).item() # 选择Q值最大的动作    
        return i_action
    
    def _update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        self.model.train(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def learn(self,
              learning_rate,
              max_epochs,
              gamma=0.9,
              epsilon_start=1.0,
              epsilon_end=0.1,
              epsilon_decay=0.995,
              max_steps_per_episode=20000,
              check_exit_freq=20):
        
        for i in range(max_epochs):
            state, _ = self.env.reset()
            for t in range(max_steps_per_episode):
                action = self._choose_action(state, epsilon)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                # state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool 
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
                if len(self.replay_buffer) >= self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
                    self._update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                
            if len(self.replay_buffer) >= self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
                self._update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

            state = next_state
            
            
        pass

    def predict(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass