
class SingleAgentEnv(ABC):
    def __init__(self, agent, world):
        self.agent = agent
        self.world = world
    
    def reset(self): 
        '''重置环境 | Reset environment'''
        self.agent.reset()
        self.world.reset()
    
    def step(self, action):
        '''执行动作 | Execute action'''
        self.agent.step(action)
        self.world.step(action)
    
    @abstractmethod
    def render(self):
        '''渲染环境 | Render environment'''
        raise NotImplementedError()