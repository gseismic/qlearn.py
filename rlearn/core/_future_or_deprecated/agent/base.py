from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """ 智能体基类 | Base agent class 
    
    Agent只负责思考, 不负责执行和奖励 | Agent only responsible for thinking, not for execution and rewards
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def reset(self):
        '''重置智能体 | Reset agent'''
        raise NotImplementedError()
    
    @abstractmethod
    def learn(self, states, actions, rewards, next_states, dones, infos):
        '''学习 | Learn
        
        Args:
            states (tuple or dict): (agent_states, world_state) 智能体状态 | Agent states
            agent_actions (dict): 动作字典 | Action dictionary
            next_states (tuple or dict): (next_agent_states, next_world_state) 下一个智能体状态 | Next agent states
            agent_rewards (dict): 奖励 | Reward
            agent_dones (dict): 是否结束 | Whether the episode is done
            agent_infos (dict): 信息 | Info
        '''
        raise NotImplementedError()

    @abstractmethod
    def act(self, agent_states, world_state):
        '''根据状态选择动作 | Choose action based on state
        
        Args:
            agent_states (dict): 智能体状态 | Agent states
            world_state (dict): 世界状态 | World state
        
        Returns:
            action (dict): 动作 | Action
        '''
        raise NotImplementedError()

    @abstractmethod
    def save(self, path):
        '''保存模型 | Save model'''
        raise NotImplementedError()

    @abstractmethod
    def load(self, path):
        '''加载模型 | Load model'''
        raise NotImplementedError()
    
    @abstractmethod
    def state_dict(self):
        '''返回模型参数 | Return model parameters'''
        raise NotImplementedError()
    
    @abstractmethod
    def load_state_dict(self, state_dict):
        '''加载模型参数 | Load model parameters'''
        raise NotImplementedError()
    
    
class AgentI(BaseAgent):
    def __init__(self, name):
        super().__init__(name)
    
    def reset(self):
        pass
    
    def learn(self, states, actions, rewards, next_states, dones, infos):
        pass
    
    
class AgentII(BaseAgent):
    def __init__(self, name):
        super().__init__(name)

    def reset(self):
        pass

    def learn(self, states, actions, rewards, next_states, dones, infos):
        pass    