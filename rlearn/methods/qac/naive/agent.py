from rlearn.methods.qac.base_agent import BaseLearnAgent
import torch
import torch.optim as optim
import numpy as np
from rlearn.methods.qac.naive.network import get_network
from rlearn.methods.utils.optimizer import get_optimizer

class QACAgent_Naive(BaseLearnAgent):
    """
    实现了一个结合 Actor-Critic 和 SARSA 的强化学习智能体。

    主要特点：
    1. 支持离散动作空间。
    2. 可以处理一维状态向量和三维图像状态。
    3. 使用 SARSA 更新来约束 Q 值估计。
    4. 可以选择使用 softmax 策略（更接近传统 Actor-Critic）或确定性策略（类似 DPG）。
    5. 支持噪声网络层，用于探索。
    6. 提供了噪声衰减机制，以逐步减少探索。

    该代理适用于各种强化学习任务，特别是在需要平衡探索和利用的情况下。
    它结合了 Actor-Critic 的稳定性和 SARSA 的样本效率。
    """
    schema = [
        dict(field='policy_net', required=False, default={
            'type': 'MLP',
            'params': {
                'hidden_dims': [128, 128],
                'activation': 'relu',
                'init_type': 'kaiming',
                'use_noisy': False,
                'factorized': True,
                'rank': 0,
                'std_init': 0.4,
                'use_softmax': True,
            }
        }, rules=dict(type='dict')),
        dict(field='optimizer', required=False, default={
            'type': 'Adam',
            'params': {
                'lr': 1e-3,
            }
        }, rules=dict(type='dict')),
        dict(field='gamma', required=False, default=0.99, rules=dict(type='float', gt=0, lt=1)),
        dict(field='device', required=False, default='cpu', rules=dict(type='str', choices=['cpu', 'cuda'])),
        dict(field='verbose_freq', required=False, default=10, rules=dict(type='int', gt=0)),
        dict(field='noise_level', required=False, default=0.0, rules=dict(type='float', ge=0)),
        dict(field='initial_noise_scale', required=False, default=0.0, rules=dict(type='float', ge=0)),
        dict(field='noise_decay', required=False, default=1.0, rules=dict(type='float', gt=0, lt=1)),
    ]

    def init(self):
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        policy_net_params = self.config['policy_net']['params']
        policy_net_params.setdefault('use_noisy', False)
        policy_net_params.setdefault('factorized', True)
        policy_net_params.setdefault('rank', 0)
        policy_net_params.setdefault('std_init', 0.4)
        
        self.ac_model = get_network(
            self.state_dim, 
            self.action_dim, 
            self.config['policy_net'],
            self.config['noise_level']
        ).to(self.config['device'])
        
        self.optimizer = get_optimizer(
            self.ac_model.parameters(),
            self.config['optimizer']
        )
        
        self.total_rewards = []
        self.episode_reward = 0
        self.noise_scale = self.config['initial_noise_scale']
        
        # Assign commonly used config values to self attributes
        self.gamma = self.config['gamma']
        self.device = self.config['device']
        self.verbose_freq = self.config['verbose_freq']
        self.noise_decay = self.config['noise_decay']
        self.use_softmax = self.config['policy_net']['params'].get('use_softmax', True)
        self.use_noisy = self.config['policy_net']['params'].get('use_noisy', False)

    def select_action(self, state, **kwargs):
        if len(state.shape) == 1:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # shape: [1, state_dim]
        elif len(state.shape) == 3:
            state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)  # shape: [1, channels, height, width]
        else:
            raise ValueError("Unsupported state dimension")
        
        probs, _ = self.ac_model(state)  # probs shape: [1, action_dim]
        if self.use_softmax:
            action = torch.multinomial(probs, 1).item()
        else:
            action = probs.argmax().item()
        
        self.next_action = action  # 存储下一个动作
        return action

    def after_step(self, state, action, reward, next_state, terminated, truncated, info, **kwargs):
        self.episode_reward += reward

        if len(state.shape) == 1:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # shape: [1, state_dim]
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)  # shape: [1, state_dim]
        elif len(state.shape) == 3:
            state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)  # shape: [1, channels, height, width]
            next_state = torch.FloatTensor(next_state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)  # shape: [1, channels, height, width]
        else:
            raise ValueError("Unsupported state dimension")
        
        probs, state_value = self.ac_model(state)  # probs shape: [1, action_dim], state_value shape: [1, 1]
        next_probs, next_state_value = self.ac_model(next_state)  # next_probs shape: [1, action_dim], next_state_value shape: [1, 1]
        
        if self.use_softmax:
            q_value = state_value + torch.log(probs[0, action])  # shape: [1, 1]
            q_next = next_state_value + torch.log(next_probs[0, self.next_action])  # shape: [1, 1]
            actor_loss = -torch.log(probs[0, action]) * td_error.detach()  # shape: [1]
        else:
            q_value = probs[0, action]  # shape: [1]
            q_next = next_probs[0, self.next_action]  # shape: [1]
            actor_loss = -probs[0, action] * td_error.detach()  # shape: [1]
        
        q_target = reward + self.gamma * q_next * (1 - terminated)  # shape: [1, 1] or [1] depending on q_next
        td_error = q_target.detach() - q_value  # shape: [1, 1] or [1] depending on q_value and q_target
        critic_loss = td_error.pow(2)  # shape: [1, 1] or [1] depending on td_error
        loss = actor_loss + critic_loss  # shape: [1] or scalar

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_noisy:
            self.ac_model.reset_noise()
        
        self.next_action = self.select_action(next_state)

    def after_episode_end(self, **kwargs):
        self.total_rewards.append(self.episode_reward)
        self.episode_reward = 0
        
        if len(self.total_rewards) % self.verbose_freq == 0:
            avg_reward = np.mean(self.total_rewards[-self.verbose_freq:])
            print(f"Average reward (last {self.verbose_freq} episodes): {avg_reward:.2f}")
        self.noise_scale *= self.noise_decay
        self.ac_model.set_noise_scale(self.noise_scale)

    def update_noise_scale(self, scale):
        self.noise_scale = scale
        self.ac_model.set_noise_scale(scale)