import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from collections import defaultdict

# 1. 五子棋环境定义
class GomokuEnv:
    def __init__(self, size=15):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # Player 1 starts

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        return self.board

    def is_legal(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == 0

    def place_piece(self, x, y):
        if self.is_legal(x, y):
            self.board[x, y] = self.current_player
            self.current_player = -self.current_player
            return True
        return False

    def check_winner(self):
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] != 0:
                    if self.check_direction(x, y, 1, 0) or \
                       self.check_direction(x, y, 0, 1) or \
                       self.check_direction(x, y, 1, 1) or \
                       self.check_direction(x, y, 1, -1):
                        return self.board[x, y]
        return 0  # No winner yet

    def check_direction(self, x, y, dx, dy):
        count = 0
        player = self.board[x, y]
        for _ in range(5):
            nx, ny = x + dx * count, y + dy * count
            if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                count += 1
            else:
                break
        return count == 5

    def step(self, action):
        x, y = action
        if self.place_piece(x, y):
            winner = self.check_winner()
            done = winner != 0 or np.all(self.board != 0)  # Check if the game is done
            reward = 1 if winner == 1 else (-1 if winner == -1 else 0)
            return self.board.copy(), reward, done
        return self.board.copy(), -10, False  # Invalid move penalty

# 2. 策略和价值网络定义
class PolicyValueNetwork(nn.Module):
    def __init__(self, size=15):
        super(PolicyValueNetwork, self).__init__()
        self.size = size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * size * size, 1024)
        self.fc_policy = nn.Linear(1024, size * size)
        self.fc_value = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.unsqueeze(1).float()  # Add channel dimension and convert to float
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.size * self.size * 128)
        x = F.relu(self.fc1(x))
        policy = F.log_softmax(self.fc_policy(x), dim=-1)
        value = torch.tanh(self.fc_value(x))
        return policy, value

# 3. MCTS 实现
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        legal_moves = self.get_legal_moves()
        return len(self.children) == len(legal_moves)

    def get_legal_moves(self):
        size = self.state.shape[0]
        return [(i, j) for i in range(size) for j in range(size) if self.state[i, j] == 0]

    def best_child(self, exploration_weight=1.):
        best_value = -float('inf')
        best_children = []
        for action, child in self.children.items():
            uct_value = child.value / (child.visits + 1e-6) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            if uct_value > best_value:
                best_value = uct_value
                best_children = [child]
            elif uct_value == best_value:
                best_children.append(child)
        return best_children[0] if best_children else None

def mcts(root_node, policy_network, n_simulations=1000):
    for _ in range(n_simulations):
        node = root_node
        state = node.state.copy()

        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            state = apply_action(node.state.copy(), node.action)

        # Expansion
        legal_moves = node.get_legal_moves()
        for move in legal_moves:
            if move not in node.children:
                next_state = apply_action(node.state.copy(), move)
                node.children[move] = MCTSNode(next_state, parent=node, action=move)
        
        # Simulation
        if node.children:
            move = np.random.choice(list(node.children.keys()))
            state = apply_action(node.state.copy(), move)
        result = simulate(state, policy_network)

        # Backpropagation
        backpropagate(node, result)
    
    return root_node.best_child(exploration_weight=0).action

def apply_action(state, action):
    x, y = action
    if 0 <= x < state.shape[0] and 0 <= y < state.shape[1] and state[x, y] == 0:
        state[x, y] = 1  # Assuming player 1 for simplicity
    return state

def simulate(state, policy_network):
    state = torch.tensor(state).unsqueeze(0)  # Convert to tensor
    policy_probs, value = policy_network(state)
    return value.item()

def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.value += result
        node = node.parent

# 4. 自我对弈生成训练数据
def self_play(env, policy_network, mcts_iterations=1000):
    data = []
    state = env.reset()
    done = False

    while not done:
        root_node = MCTSNode(state)
        action = mcts(root_node, policy_network, n_simulations=mcts_iterations)
        next_state, reward, done = env.step(action)
        data.append((state, action, reward))
        state = next_state

    return data

def generate_training_data(env, policy_network, n_games=100, mcts_iterations=1000):
    all_data = []
    for _ in range(n_games):
        game_data = self_play(env, policy_network, mcts_iterations)
        all_data.extend(game_data)
    return all_data

# 5. 训练神经网络
def train_policy_value_network(network, data, epochs=10, batch_size=64, lr=0.001):
    optimizer = optim.Adam(network.parameters(), lr=lr)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    for epoch in range(epochs):
        np.random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            states, actions, rewards = zip(*batch)

            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(np.array(actions), dtype=torch.long)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32)

            optimizer.zero_grad()
            policy_probs, values = network(states)
            policy_loss = criterion_policy(policy_probs, actions)
            value_loss = criterion_value(values.squeeze(), rewards)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# 6. 使用训练后的网络进行对弈
def play_game(env, policy_network, mcts_iterations=1000):
    state = env.reset()
    done = False
    while not done:
        root_node = MCTSNode(state)
        action = mcts(root_node, policy_network, n_simulations=mcts_iterations)
        next_state, reward, done = env.step(action)
        env.board = next_state  # Update board state for the next turn

        # Print the board state (for debugging)
        print(env.board)
        
        if done:
            print(f"Game Over! Reward: {reward}")
            return reward

# 主执行流程
if __name__ == "__main__":
    # Initialize environment and network
    env = GomokuEnv(size=15)
    policy_network = PolicyValueNetwork(size=15)
    policy_network.train()

    # Generate and train
    training_data = generate_training_data(env, policy_network, n_games=10, mcts_iterations=1000)
    train_policy_value_network(policy_network, training_data, epochs=10, batch_size=64, lr=0.001)

    # Play a game
    play_game(env, policy_network, mcts_iterations=1000)

