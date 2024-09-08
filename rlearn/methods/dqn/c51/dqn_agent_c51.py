import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ....core.agent import Agent
from ....utils.replay_buffer import ReplayBuffer
from .network import C51Network
from ....utils.config import Config

DEFAULT_CONFIG = {
    'env': {
        'state_dim': None,
        'action_dim': None,
    },
    'algorithm': {
        'num_atoms': 51,
        'v_min': -10,
        'v_max': 10,
        'gamma': 0.99,
        'use_double_dqn': True,
        'use_dueling': False,
        'use_noisy_nets': False
    },
    'network': {
        'hidden_sizes': [128, 128],
    },
    'training': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'target_update_freq': 100,
        'grad_clip': 10.0,
        'optimizer': 'adam',
        'loss_fn': 'cross_entropy'
    },
    'exploration': {
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995
    },
    'memory': {
        'buffer_size': 10000,
        'prioritized_replay': False,
        'alpha': 0.6,
        'beta': 0.4,
        'beta_increment': 0.001
    },
    'device': 'cpu'
}

class DQNAgent_C51(Agent):
    def __init__(self, name, env, state_dim, action_dim, num_atoms=51, v_min=-10, v_max=10, 
                 learning_rate=0.001, gamma=0.99, batch_size=32, buffer_size=10000, 
                 device='cpu', *args, **kwargs):
        super(DQNAgent_C51, self).__init__(name=name, env=env, *args, **kwargs)
        self.config = Config() 
        self.config.set('env.state_dim', state_dim, is_int=True, gt=0)
        self.config.set('env.action_dim', action_dim, is_int=True, gt=0)
        self.config.set('algorithm.num_atoms', num_atoms, is_int=True, gt=0)
        self.config.set('algorithm.v_min', v_min, is_float=True)
        self.config.set('algorithm.v_max', v_max, is_float=True, gt='algorithm.v_min')
        self.config.set('algorithm.gamma', gamma, is_float=True, ge=0, le=1)
        self.config.set('algorithm.batch_size', batch_size, is_int=True, gt=16)
        self.config.set('device', device, is_str=True, in_values=['cpu', 'cuda'])
        # self.config.set('algorithm.target_update_freq', target_update_freq, is_int=True, ge=0)
        # self.config.set('algorithm.grad_clip', grad_clip, is_float=True, ge=0)
        # self.config.set('algorithm.optimizer', optimizer)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = torch.device(device) if isinstance(device, str) else device
        
        self.z = torch.linspace(v_min, v_max, num_atoms)        
        self.policy_net = C51Network(state_dim, action_dim, num_atoms, v_min, v_max).to(self.device)
        self.target_net = C51Network(state_dim, action_dim, num_atoms, v_min, v_max).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def choose_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            distribution = self.policy_net(state)
            expected_value = (distribution * self.z.unsqueeze(0).unsqueeze(0)).sum(2)
            action = expected_value.argmax(1).item()
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        # 将样本转换为张量 | Convert samples to tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)  # shape: (batch_size, state_dim)
        action_batch = torch.LongTensor(action_batch).to(self.device)  # shape: (batch_size,)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)  # shape: (batch_size,)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)  # shape: (batch_size, state_dim)
        done_batch = torch.FloatTensor(done_batch).to(self.device)  # shape: (batch_size,)

        # 计算下一个状态的分布 | Calculate the distribution of the next state
        with torch.no_grad():
            next_distribution = self.target_net(next_state_batch) # s
            next_q = (next_distribution * self.z.unsqueeze(0).unsqueeze(0)).sum(2)
            # XXX: next_q.argmax(1) or epsilon-greedy?
            next_action = next_q.argmax(1)
            next_distribution = next_distribution[range(self.batch_size), next_action]

            # 计算目标分布 | Calculate target distribution
            Tz = reward_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * self.gamma * self.z.unsqueeze(0)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            target_distribution = torch.zeros_like(next_distribution) # shape: (batch_size, num_atoms)
            # 创建一个偏移量张量，用于正确索引目标分布 | Create an offset tensor for correct indexing of target distribution
            # >>> batch_size = 3
            # >>> num_atom=5
            # >>> A = torch.linspace(0, (batch_size-1)*num_atom, batch_size)
            # >>> A
            # tensor([ 0.,  5., 10.])
            # >>> A.long()
            # tensor([ 0,  5, 10])
            # >>> A.long().unsqueeze(1)
            # tensor([[ 0],
            #         [ 5],
            #         [10]])
            # >>> A.long().unsqueeze(1).expand(batch_size, num_atom)
            # tensor([[ 0,  0,  0,  0,  0],
            #         [ 5,  5,  5,  5,  5],
            #         [10, 10, 10, 10, 10]])
            offset = torch.linspace(0, 
                                    (self.batch_size - 1) * self.num_atoms, 
                                    self.batch_size
                                    ).long().unsqueeze(1).expand(self.batch_size, self.num_atoms)
            # offset shape: (batch_size, num_atoms)
            # 使用index_add_方法更新目标分布 | Use index_add_ method to update target distribution
            # 对于下界l，添加(u - b)的权重 | For lower bound l, add the weight of (u - b)
            target_distribution.view(-1).index_add_(0, (l + offset).view(-1), (next_distribution * (u.float() - b)).view(-1))
            # 对于上界u，添加(b - l)的权重 | For upper bound u, add the weight of (b - l)
            target_distribution.view(-1).index_add_(0, (u + offset).view(-1), (next_distribution * (b - l.float())).view(-1))

        # 计算当前分布 | Calculate current distribution
        current_distribution = self.policy_net(state_batch)
        current_distribution = current_distribution[range(self.batch_size), action_batch]

        # 计算损失 | Calculate loss
        loss = -(target_distribution * torch.log(current_distribution + 1e-8)).sum(1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self, num_episodes, max_steps_per_episode=1000):
        """
        Note:
            agent = DQNAgent_C51(...)
            agent.learn(num_episodes=100, max_steps_per_episode=1000)
            agent.learn(num_episodes=200, max_steps_per_episode=1000)
        """
        reward_list = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            for step in range(max_steps_per_episode):
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.replay_buffer.push(state, action, reward, next_state, done)
                self.update()

                state = next_state
                episode_reward += reward

                if done:
                    break

            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            reward_list.append(episode_reward)  
            print(f"Episode {episode}, Total Reward: {episode_reward}")

        return reward_list
    
    def predict(self, state, *args, **kwargs):
        return super().predict(state, *args, **kwargs)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
