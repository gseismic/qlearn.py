import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gym

# pip install gym==0.21.0  # 安装特定版本的 Gym

# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化环境
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 初始化Q网络和目标网络
q_network = QNetwork(state_dim, action_dim)
target_network = QNetwork(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())

optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
replay_buffer = deque(maxlen=10000)
batch_size = 64
gamma = 0.99
epsilon = 0.1
epsilon_decay = 0.995
epsilon_min = 0.01
target_update_interval = 1000

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_network(state_tensor)
            return q_values.argmax().item()

def optimize_model():
    if len(replay_buffer) < batch_size:
        return
    
    # 随机抽取经验
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    
    # 计算Q值和目标Q值
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0]
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))
    
    # 计算损失
    loss = nn.MSELoss()(q_values, target_q_values)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 主训练循环
num_episodes = 1000
for episode in range(num_episodes):
    state, _ = env.reset()
    print(f'{state=}')
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    for t in range(200):  # 假设最大步数为200
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        
        optimize_model()
        
        if done:
            break
    
    if episode % target_update_interval == 0:
        target_network.load_state_dict(q_network.state_dict())

