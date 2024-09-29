import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'leaky_relu': nn.LeakyReLU
}

OPTIMIZERS = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop
}

class DefaultActorCriticModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=None, activation=None):
        super(DefaultActorCriticModel, self).__init__()
        hidden_sizes = hidden_sizes or [128]
        activation = activation or nn.ReLU
        # 构建 actor 网络
        actor_layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            actor_layers.append(nn.Linear(input_dim, hidden_size))
            actor_layers.append(activation())
            input_dim = hidden_size
        actor_layers.append(nn.Linear(input_dim, action_dim))
        actor_layers.append(nn.Softmax(dim=-1))
        self.actor = nn.Sequential(*actor_layers)
        
        # 构建 critic 网络
        critic_layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            critic_layers.append(nn.Linear(input_dim, hidden_size))
            critic_layers.append(activation())
            input_dim = hidden_size
        critic_layers.append(nn.Linear(input_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

class QACAgent:
    """基于TD约束的actor-critic QAC算法
    
    actor网络: 行动决策
    critic网络: 价值判断, 用TD方法约束q(s_t, a_t)
    """
    
    def __init__(self, env, config=None):
        self.env = env
        self.config = config or {}
        self.build_model()
    
    def build_model(self):
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        model_class = self.config.get('model_class', DefaultActorCriticModel)
        model_kwargs = self.config.get('model_kwargs', {})
        
        if model_class == DefaultActorCriticModel:
            hidden_sizes = model_kwargs.get('hidden_sizes', [128])
            if isinstance(hidden_sizes, str):
                hidden_sizes = [int(size) for size in hidden_sizes.split(',')]
            activation = ACTIVATIONS[model_kwargs.get('activation', 'relu')]
            self.model = model_class(state_dim, action_dim, hidden_sizes, activation)
        else:
            self.model = model_class(state_dim, action_dim, **model_kwargs)
        
        optimizer_class = OPTIMIZERS[self.config.get('optimizer', 'adam')]
        optimizer_kwargs = self.config.get('optimizer_kwargs', {'lr': 0.001})
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.model(state_tensor)
        action = np.random.choice(len(action_probs.squeeze()), p=action_probs.detach().numpy().squeeze())
        return action

    def step(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # 计算TD目标
        _, value = self.model(state_tensor)
        _, next_value = self.model(next_state_tensor)
        td_target = reward + (1 - done) * next_value.item()
        td_error = td_target - value.item()

        # 更新模型
        self.optimizer.zero_grad()
        action_probs, value = self.model(state_tensor)
        actor_loss = -torch.log(action_probs[0, action]) * td_error
        critic_loss = torch.nn.functional.mse_loss(value, torch.tensor([[td_target]]))
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()

        return td_error

    def learn(self, num_episodes, verbose_freq=10):
        max_steps = self.config.get('max_steps', 1000)
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                total_reward += reward

                self.step(state, action, reward, next_state, done)

                state = next_state
                if done or truncated:
                    break
            
            if (episode + 1) % verbose_freq == 0:
                print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    def predict(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = self.model(state_tensor)
            return action_probs.argmax().item()

def main():
    env = gym.make("CartPole-v1")
    
    # 使用混合类型配置
    config = {
        'model_class': DefaultActorCriticModel,
        'model_kwargs': {
            # 'hidden_sizes': [64, 64],
            'hidden_sizes': [16, 16],
            'activation': 'relu',
        },
        'optimizer': 'adam',
        'optimizer_kwargs': {'lr': 0.0003},
        'max_steps': 2000
    }
    
    agent = QACAgent(env, config=config)
    agent.learn(num_episodes=5000, verbose_freq=10)

    test_episodes = 10
    for episode in range(test_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.predict(state)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if truncated:
                break
        print(f"Test Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()