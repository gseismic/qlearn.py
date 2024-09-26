import torch
import numpy as np
from cfgdict import make_config
from rlearn.logger import user_logger
from rlearn.methods.utils.monitor import RewardMonitor

class BaseDQNAgent:

    def __init__(self, env, config=None, logger=None):
        self.env = env
        self.logger = logger or user_logger
        self.config = make_config(config, schema=self.schema,
                                  logger=self.logger,
                                  to_dict=True,
                                  to_dict_flatten=True)
        self.monitor = None
        if self.config['device'] == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.logger.info(f'Config: {self.config}')
        # self.init_networks()

    def init_networks(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def select_action(self, state):
        raise NotImplementedError("This method should be overridden by subclasses")

    def step_update(self, batch):
        if len(self.memory) < self.config['batch_size']:
            return
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.config['gamma'] * next_q_values * (1 - done_batch)
        
        loss = torch.nn.functional.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps_done % self.config['target_update'] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self, 
              num_episodes, 
              max_step_per_episode=None, 
              max_total_steps=None, 
              target_reward=None, 
              seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.monitor = RewardMonitor(
            max_step_per_episode=max_step_per_episode,
            max_total_steps=max_total_steps,
            target_reward=target_reward
        )
        
        for episode in range(num_episodes):
            self.monitor.reset()
            state, _ = self.env.reset()
            
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                self.memory.push(state, action, reward, next_state, done)
                self.step_update(self.memory.sample(self.config['batch_size']))
                
                self.monitor.update(next_state, reward, terminated, truncated, info)
                state = next_state
                self.steps_done += 1
                
                exit_episode, exit_learning, exit_learning_msg = self.monitor.check_exit_conditions()
                if exit_episode:
                    break
                
                if exit_learning:
                    self.logger.info(exit_learning_msg)
                    return

            if (episode + 1) % self.config['verbose_freq'] == 0:
                self.logger.info(f"Episode {episode+1}/{num_episodes}, Total Reward: {self.monitor.total_reward}")

    def save(self, file_path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
