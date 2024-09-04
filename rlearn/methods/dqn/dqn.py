
"""
```
Initialize Q-network with random weights
Initialize target Q-network with the same weights as Q-network
Initialize experience replay buffer

for each episode:
    Initialize state s
    for each step in the episode:
        with probability ε:
            Select a random action a
        otherwise:
            Select action a = argmax_a Q(s, a)  # Greedy action selection

        Execute action a, observe reward r and next state s'

        Store experience (s, a, r, s') in experience replay buffer

        Sample a random minibatch of experiences from the replay buffer

        for each experience (s, a, r, s') in minibatch:
            Compute target value:
                target = r + γ * max_a' Q_target(s', a')
            Compute loss:
                loss = (Q(s, a) - target)^2

        Perform gradient descent on the loss to update Q-network weights

        Every C steps:
            Update target Q-network weights with Q-network weights
```
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, hidden_sizes, action_size, activation=nn.ReLU, output_activation=nn.Identity):
        super(DQN, self).__init__()
        layers = [] 
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(activation())
        layers.append(nn.Linear(hidden_sizes[-2], hidden_sizes[-1]))
        layers.append(activation())
        self.layers = nn.Sequential(*layers)
        self.fc3 = nn.Linear(hidden_sizes[-1], action_size)
        self.activation = activation
        self.output_activation = output_activation

    def forward(self, x):
        x = self.layers(x)
        return self.output_activation(self.fc3(x))

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_dqn(env, num_episodes, update_frequency=100):
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

            if episode % update_frequency == 0:
                agent.update_target_network()

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    return agent
"""
