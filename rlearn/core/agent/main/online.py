from abc import abstractmethod
from rlearn.methods.utils.monitor import get_monitor
from rlearn.core.agent.main.base import BaseAgent
import os

# RequiredField
# Field('monitor', required=True, help='Monitor config')
# Field('verbose_freq', required=False, default=10, help='Frequency of logging')
class OnlineAgent(BaseAgent):
    schema = [
        # Field('monitor', type=dict, default=None, help='Monitor config'),
        # Field('verbose_freq', type=int, default=10, help='Frequency of logging')
    ]

    def __init__(self, env, config=None, logger=None, seed=None):
        super().__init__(env, config, logger, seed)
        monitor_cfg = self.config.get('monitor', {})
        self.monitor = get_monitor(
            monitor_cfg.get('type'),
            **monitor_cfg.get('config', {})
        )
        self.verbose_freq = self.config.get('verbose_freq', 10)
        self.checkpoint_dir = self.config.get('checkpoint_dir', None)
        self.checkpoint_freq = self.config.get('checkpoint_freq', None)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.init()
    
    def init(self):
        pass
        
    def before_learn(self):
        pass
    
    def before_episode_start(self, state, info, **kwargs):
        pass
        
    @abstractmethod
    def select_action(self, state, **kwargs):
        pass

    @abstractmethod
    def after_env_step(self, state, action, reward, next_state, terminated, truncated, info, **kwargs):
        pass
    
    @abstractmethod
    def after_episode_end(self, **kwargs):
        pass
    
    def after_learn_done(self, exit_info):
        pass
    
    def learn(self, num_episodes, **kwargs):
        self.logger.info(f"Start learning")

        self.before_learn()
        should_exit_learning = False
        for episode_idx in range(num_episodes):
            state, _ = self.env.reset()
            self.monitor.before_episode_start()
            self.before_episode_start(state, info, **kwargs)

            should_exit_episode = False
            while not should_exit_episode:
                action = self.select_action(state, **kwargs)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                self.monitor.after_env_step(next_state, reward, terminated, truncated, info)
                self.after_env_step(state, action, reward, next_state, terminated, truncated, info, **kwargs)
                
                state = next_state
                should_exit_episode, should_exit_learning, (exit_learning_code, exit_learning_msg) = self.monitor.check_exit_conditions()

                should_exit_episode |= terminated or truncated or should_exit_learning
            
            self.monitor.after_episode_end()
            self.after_episode_end(**kwargs)
                
            if (episode_idx + 1) % self.verbose_freq == 0:
                self.logger.info(f"Episode {episode_idx+1}/{num_episodes}, Episode Reward: {self.monitor.episode_reward}")
   
            if should_exit_learning:
                if exit_learning_code == 0:
                    self.logger.info(exit_learning_msg)
                elif exit_learning_code > 0:
                    self.logger.warning(exit_learning_msg)
                else:
                    raise ValueError(f"Invalid exit learning code: {exit_learning_code}")
                break
            
            if self.checkpoint_dir is not None and (self.checkpoint_freq is None or (episode_idx + 1) % self.checkpoint_freq == 0):
                self.save_checkpoint(episode_idx)
            if episode_idx == num_episodes - 1:
                self.logger.warning(f"Reached the maximum number of episodes: {num_episodes}")
        
        exit_info = self.monitor.get_exit_info()
        self.logger.info(f"Exit Info: {exit_info}")
        self.after_learn_done(exit_info)
        return exit_info

    def save_checkpoint(self, episode_idx):
        pass
