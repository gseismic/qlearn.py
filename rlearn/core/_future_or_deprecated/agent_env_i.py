from abc import ABC, abstractmethod

class AgentEnvI(ABC):   

    def __init__(self, agent, world):
        self.agent = agent
        self.world = world
    
    def reset(self): 
        '''重置环境 | Reset environment'''
        self.agent.reset()
        self.world.reset()
    
    def step(self):
        '''执行动作 | Execute action'''
        agent_action = self.agent.act(self.agent.state_dict(), self.world.state_dict())
        (next_agent_state, next_world_state), agent_reward, agent_done, agent_info = self.world.step(self.agent.state_dict(), agent_action)
        self.agent.learn(self.agent.state_dict(), agent_action, next_agent_state, agent_reward, agent_done, agent_info)
        return (self.agent.state_dict(), self.world.state_dict()), agent_action, (next_agent_state, next_world_state), agent_reward, agent_done, agent_info
    
    def render(self):
        '''渲染环境 | Render environment'''
        self.agent.render()
        self.world.render()
