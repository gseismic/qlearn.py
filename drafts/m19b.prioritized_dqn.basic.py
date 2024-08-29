import numpy as np
import random
from collections import namedtuple

# 定义经验的命名元组
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.data_pointer = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, value):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-5  # 用于避免优先级为零
        self.max_priority = 1.0  # 初始化最大优先级
        self.capacity = capacity

    def add(self, experience):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, priority, data) = self.tree.get_leaf(s)
            
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total_priority()
        is_weights = np.power(self.tree.size * sampling_probabilities, -beta)
        is_weights /= is_weights.max()  # 标准化

        return batch, idxs, is_weights

    def update_priorities(self, idxs, priorities):
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority ** self.alpha)
            self.max_priority = max(self.max_priority, priority)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
def compute_td_error(dqn, target_dqn, batch, gamma):
    states, actions, rewards, next_states, dones = batch
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_dqn(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    td_error = torch.abs(q_values - target_q_values)
    return td_error.cpu().numpy()

def train_dqn(dqn, target_dqn, memory, optimizer, batch_size, gamma, beta):
    batch, idxs, is_weights = memory.sample(batch_size, beta)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    is_weights = torch.tensor(is_weights, dtype=torch.float32)

    q_values = dqn(states).gather(1, actions)
    with torch.no_grad():
        next_q_values = target_dqn(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards.unsqueeze(1) + gamma * next_q_values * (1 - dones.unsqueeze(1))

    loss = (is_weights * F.mse_loss(q_values, target_q_values, reduction='none')).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    td_errors = compute_td_error(dqn, target_dqn, (states, actions.squeeze(), rewards, next_states, dones), gamma)
    memory.update_priorities(idxs, td_errors)
# Hyperparameters
state_size = 4  # 根据环境设置
action_size = 2  # 根据环境设置
gamma = 0.99
alpha = 0.6
beta_start = 0.4
beta_frames = 1000
batch_size = 32
memory_capacity = 10000
lr = 1e-3
num_episodes = 500

dqn = DQN(state_size, action_size)
target_dqn = DQN(state_size, action_size)
target_dqn.load_state_dict(dqn.state_dict())
optimizer = optim.Adam(dqn.parameters(), lr=lr)

memory = PrioritizedReplayBuffer(memory_capacity, alpha)

# Main loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        # 动作选择（e.g., epsilon-greedy）
        action = env.action_space.sample()  # 这里使用随机策略，实际中应使用贪婪策略

        # 环境交互
        next_state, reward, done, _ = env.step(action)
        memory.add(Experience(state, action, reward, next_state, done))

        state = next_state

        # 检查是否可以训练
        if memory.tree.size > batch_size:
            beta = min(1.0, beta_start + episode / beta_frames * (1.0 - beta_start))
            train_dqn(dqn, target_dqn, memory, optimizer, batch_size, gamma, beta)

    # 更新目标网络
    if episode % 10 == 0:
        target_dqn.load_state_dict(dqn.state_dict())

