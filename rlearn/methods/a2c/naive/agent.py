import numpy as np
import torch
import torch.optim as optim
from .network import ActorCritic

"""
初始化:
    策略网络 π(a|s) 参数 θ
    价值网络 V(s) 参数 w
    学习率 α_θ, α_w
    折扣因子 γ

对于每个回合:
    初始化状态 s
    对于 t = 0, 1, 2, ..., T-1:
        根据策略π(a|s)选择动作 a
        执行动作 a, 观察奖励 r 和新状态 s'
        计算 TD 误差: δ = r + γV(s') - V(s)
        更新策略参数: θ = θ + α_θ ∇θ log π(a|s) δ
        更新价值参数: w = w + α_w δ ∇w V(s)
        s = s'
    直到 s 是终止状态
---
Initialize:
    Policy network π(a|s) parameters θ
    Value network V(s) parameters w
    Learning rates α_θ, α_w
    Discount factor γ

For each episode:
    Initialize state s
    For t = 0, 1, 2, ..., T-1:
        Choose action a according to policy π(a|s)
        Execute action a, observe reward r and new state s'
        Compute TD error: δ = r + γV(s') - V(s)
        Update policy parameters: θ = θ + α_θ ∇θ log π(a|s) δ
        Update value parameters: w = w + α_w δ ∇w V(s)
        s = s'
    Until s is a terminal state
"""

class A2CAgent_Naive:
    DEFAULT_CONFIG = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'device': 'cpu'
    }
    def __init__(self, env, config=None):
        self.env = env
        config = config or {}
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.input_dim = np.prod(env.observation_space.shape)
        self.n_actions = env.action_space.n  
        if self.config['device'] == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.model = ActorCritic(self.input_dim, self.n_actions).to(self.device)
        self.gamma = self.config['gamma']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
    
    def choose_action(self, state):
        # Choose an action based on the policy
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, _ = self.model(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, state, action, reward, next_state, done):
        # Convert to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([int(done)]).to(self.device)
        
        # Compute advantage
        _, current_value = self.model(state)
        _, next_value = self.model(next_state)
        delta = reward + self.gamma * next_value * (1 - done) - current_value
        
        # Compute actor and critic losses
        action_probs, current_value = self.model(state)
        action_log_probs = torch.log(action_probs.squeeze(0))
        actor_loss = -action_log_probs[action] * delta.detach()
        critic_loss = delta.pow(2)
        
        # Compute total loss and optimize
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def learn(self, num_episodes, seed=None):
        for episode in range(num_episodes):
            state, _ = self.env.reset(seed=seed)
            done = False
            total_reward = 0
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}, Total Reward: {total_reward}")