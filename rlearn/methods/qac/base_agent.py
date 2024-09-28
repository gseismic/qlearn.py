from abc import ABC, abstractmethod
from cfgdict import make_config
from rlearn.logger import user_logger
from rlearn.utils.seed import seed_all
from rlearn.methods.utils.monitor import get_monitor
import torch
class BaseLearnAgent(ABC):
    schema = []

    def __init__(self, env, config=None, logger=None, seed=None):
        self.env = env
        self.logger = logger or user_logger
        self.config = make_config(config, schema=self.schema,
                                  logger=self.logger,
                                  to_dict=True,
                                  to_dict_flatten=False)
        self.try_seed_all(seed)
        monitor_cfg = self.config.get('monitor', {})
        self.monitor = get_monitor(
            monitor_cfg.get('type', 'reward'),
            **monitor_cfg.get('config', {})
        )
        self.device = self.config.get('device', 'cpu') if torch.cuda.is_available() else 'cpu'
        self.verbose_freq = self.config.get('verbose_freq', 10)
        self.init()
    
    def try_seed_all(self, seed):
        if seed is None:
            return
        seed_all(seed)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
    
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
    def after_step(self, state, action, reward, next_state, terminated, truncated, info, **kwargs):
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
                
                self.monitor.after_step_env(next_state, reward, terminated, truncated, info)
                self.after_step(state, action, reward, next_state, terminated, truncated, info, **kwargs)
                
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
            
            if episode_idx == num_episodes - 1:
                self.logger.warning(f"Reached the maximum number of episodes: {num_episodes}")
        
        exit_info = self.monitor.get_exit_info()
        self.logger.info(f"Exit Info: {exit_info}")
        self.after_learn_done(exit_info)
        return exit_info
