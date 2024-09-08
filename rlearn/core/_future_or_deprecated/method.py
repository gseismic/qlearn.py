import numpy as np
from abc import ABC, abstractmethod

class RLOptimizer(ABC):
    def __init__(self, env, learning_rate=0.01, discount_factor=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_action(self, state):
        pass

class OfflineRLOptimizer(RLOptimizer):
    def __init__(self, env, learning_rate=0.01, discount_factor=0.99, batch_size=32):
        super().__init__(env, learning_rate, discount_factor)
        self.batch_size = batch_size

    @abstractmethod
    def train_on_batch(self, batch):
        pass

class OnlineRLOptimizer(RLOptimizer):
    def __init__(self, env, learning_rate=0.01, discount_factor=0.99):
        super().__init__(env, learning_rate, discount_factor)

    @abstractmethod
    def update_online(self, state, action, reward, next_state, done):
        pass

class PPO(OfflineRLOptimizer):
    def __init__(self, env, learning_rate=0.001, discount_factor=0.99, batch_size=64, clip_ratio=0.2, epochs=10):
        super().__init__(env, learning_rate, discount_factor, batch_size)
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.actor = None  # 需要初始化actor网络
        self.critic = None  # 需要初始化critic网络
        self.optimizer = None  # 需要初始化优化器

    def update(self, states, actions, old_log_probs, rewards, advantages):
        # PPO更新逻辑
        for _ in range(self.epochs):
            # 计算新的动作概率和值函数
            new_log_probs, state_values, entropy = self.evaluate(states, actions)
            
            # 计算比率
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # 计算surrogate损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            # 计算actor损失
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算critic损失
            critic_loss = nn.MSELoss()(state_values, rewards)
            
            # 计算总损失
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
            
            # 执行优化步骤
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_action(self, state):
        # 获取PPO动作
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def train_on_batch(self, batch):
        # 在批次上训练PPO
        states, actions, old_log_probs, rewards, advantages = batch
        
        # 将数据转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        advantages = torch.FloatTensor(advantages)
        
        self.update(states, actions, old_log_probs, rewards, advantages)
        
        return self.evaluate(states, actions)[0].mean().item()  # 返回平均log概率作为损失

    def evaluate(self, states, actions):
        # 评估动作
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        state_values = self.critic(states).squeeze()
        dist_entropy = dist.entropy()
        
        return action_log_probs, state_values, dist_entropy
    
    
class QLearning(OnlineRLOptimizer):
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        super().__init__(env, learning_rate, discount_factor)
        self.epsilon = epsilon
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def update(self, state, action, reward, next_state):
        # Q-learning更新逻辑 | Q-learning update logic
        self.Q[state, action] += self.learning_rate * (
            reward + self.discount_factor * np.max(self.Q[next_state]) - self.Q[state, action]
        )

    def get_action(self, state):
        # ε-贪婪策略 | ε-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state])

    def update_online(self, state, action, reward, next_state, done):
        # 在线更新Q值 | Online update Q value
        self.update(state, action, reward, next_state)
        return self.get_action(next_state)

class ValueIteration(RLOptimizer):
    def __init__(self, env, discount_factor=0.99, theta=1e-6):
        super().__init__(env, discount_factor=discount_factor)
        self.theta = theta
        self.V = np.zeros(env.observation_space.n)

    def update(self):
        # 状态值迭代算法 | State value iteration algorithm
        while True:
            delta = 0
            for s in range(self.env.observation_space.n):
                v = self.V[s]
                self.V[s] = max([sum([p * (r + self.discount_factor * self.V[s_]) 
                                 for p, s_, r, _ in self.env.P[s][a]])
                                 for a in range(self.env.action_space.n)])
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta:
                break

    def get_action(self, state):
        # 根据状态值选择最佳动作 | Select the best action based on state value
        return np.argmax([sum([p * (r + self.discount_factor * self.V[s_]) 
                          for p, s_, r, _ in self.env.P[state][a]])
                          for a in range(self.env.action_space.n)])

class TruncatedPolicyIteration(RLOptimizer):
    def __init__(self, env, discount_factor=0.99, max_iterations=10, theta=1e-6):
        super().__init__(env, discount_factor=discount_factor)
        self.max_iterations = max_iterations
        self.theta = theta
        self.V = np.zeros(env.observation_space.n)
        self.policy = np.zeros(env.observation_space.n, dtype=int)

    def update(self):
        # 截断策略迭代算法 | Truncated policy iteration algorithm
        for _ in range(self.max_iterations):
            self._policy_evaluation()
            self._policy_improvement()

    def _policy_evaluation(self):
        for _ in range(self.max_iterations):
            delta = 0
            for s in range(self.env.observation_space.n):
                v = self.V[s]
                a = self.policy[s]
                self.V[s] = sum([p * (r + self.discount_factor * self.V[s_]) 
                                 for p, s_, r, _ in self.env.P[s][a]])
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta:
                break

    def _policy_improvement(self):
        policy_stable = True
        for s in range(self.env.observation_space.n):
            old_action = self.policy[s]
            self.policy[s] = np.argmax([sum([p * (r + self.discount_factor * self.V[s_]) 
                                        for p, s_, r, _ in self.env.P[s][a]])
                                        for a in range(self.env.action_space.n)])
            if old_action != self.policy[s]:
                policy_stable = False
        return policy_stable

    def get_action(self, state):
        return self.policy[state]

    def update(self, states, actions, old_log_probs, rewards, advantages):
        # PPO更新逻辑 | PPO update logic
        pass

    def get_action(self, state):
        # 获取PPO动作 | Get PPO action
        pass

    def train_on_batch(self, batch):
        # 在批次上训练PPO | Train PPO on batch
        pass

class SARSA(OnlineRLOptimizer):
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        super().__init__(env, learning_rate, discount_factor)
        self.epsilon = epsilon
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def update(self, state, action, reward, next_state, next_action):
        # SARSA更新逻辑 | SARSA update logic
        self.Q[state, action] += self.learning_rate * (
            reward + self.discount_factor * self.Q[next_state, next_action] - self.Q[state, action]
        )

    def get_action(self, state):
        # ε-贪婪策略 | ε-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state])

    def update_online(self, state, action, reward, next_state, done):
        # 在线更新Q值 | Online update Q value
        next_action = self.get_action(next_state)
        self.update(state, action, reward, next_state, next_action)
        return next_action

