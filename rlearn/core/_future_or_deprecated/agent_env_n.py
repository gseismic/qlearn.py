from collections import OrderedDict
from abc import ABC, abstractmethod 


class AgentEnvN(ABC):   
    def __init__(self, agents, world):
        self.agents = agents
        self.world = world
    
    def reset(self): 
        '''重置环境 | Reset environment'''
        for agent in self.agents:
            agent.reset()
        self.world.reset()
    
    def step(self):
        '''执行动作 | Execute action
        
        所有智能体【无沟通的】按各自策略【同时】执行动作，然后【一次性】更新世界状态 | 
        All agents execute actions simultaneously without communication based on their own strategies, 
        and then the world state is updated once.
        
        比如:3个人(A,B,C)玩扑克游戏,轮流出牌,A->B->C,此时需要把动作拆分3次 | 
        For example, in a card game with 3 players (A, B, C), the action is executed in turns: A->B->C. 
        The action needs to be split into 3 times:
            [A_action, None, None]
            [None, B_action, None]
            [None, None, C_action]

        Returns:
            agent_states (dict): 智能体状态 | Agent states
            agent_actions (dict): 动作字典 | Action dictionary
            next_agent_states (dict): 下一个智能体状态 | Next agent states
            next_world_state (dict): 下一个世界状态 | Next world state
            agent_rewards (dict): 奖励 | Rewards
            agent_ends (dict): 是否结束 | Whether the episode is done
            end_infos (dict): 信息 | Info
        '''
        # 获取智能体状态 | Get agent states
        agent_states = {}
        world_state = {}
        for agent in self.agents:
            agent_states[agent.name] = agent.state_dict() 
        # 获取世界状态 | Get world state
        world_state = self.world.state_dict()
        
        agent_actions = OrderedDict()
        for agent in self.agents:
            agent_actions[agent.name] = agent.act(agent_states, world_state)
        (next_agent_states, next_world_state), agent_rewards, agent_ends, end_infos = self.world.step(agent_states, agent_actions)   
        for name, agent in agent_ends.items():
            agent.learn(
                (agent_states, world_state), 
                agent_actions,
                (next_agent_states, next_world_state), 
                agent_rewards, # 一些奖励是基于下一个世界/Agents状态的，所以放在这里
                agent_dones,
                end_infos
            )
        return (agent_states, world_state), agent_actions, (next_agent_states, next_world_state), agent_rewards, agent_dones, end_infos
    
    def render(self):
        '''渲染环境 | Render environment'''
        for agent in self.agents:
            agent.render()
        self.world.render()
