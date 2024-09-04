from abc import ABC, abstractmethod

class BaseWorld(ABC):
    
    @abstractmethod
    def reset(self):
        '''重置环境 | Reset environment'''
        raise NotImplementedError()
    
    @abstractmethod
    def step(self, agent_states, agent_actions):
        '''执行动作 | Execute action
        
        Args:
            agent_states (dict): 智能体状态 | Agent states
            agent_actions (dict): 动作字典 | Action dictionary
        
        Returns:
            next_agent_states (dict): 下一个智能体状态 | Next agent states
            next_world_state (dict): 下一个世界状态 | Next world state
            agent_rewards (dict): 奖励 | Reward
            agent_dones (dict): 是否结束 | Whether the episode is done
            agent_infos (dict): 信息 | Info
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def render(self):
        '''渲染环境 | Render environment'''
        raise NotImplementedError()
    
    @abstractmethod
    def close(self):
        '''关闭环境 | Close environment'''
        raise NotImplementedError()
    
class BaseWorldI(BaseWorld):
    def __init__(self, state_dict):
        self.state_dict = state_dict
    
    def reset(self):
        pass
    
    def step(self, action):
        pass    
    
class BaseWorldN(BaseWorld):
    def __init__(self, state_dict):
        self.state_dict = state_dict
    
    def reset(self):
        pass
    
    def step(self, action):
        pass
