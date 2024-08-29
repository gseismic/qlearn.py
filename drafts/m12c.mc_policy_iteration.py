import numpy as np
import random
from collections import defaultdict

class GridWorldEnv:
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4)):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.reset()
        
    def reset(self):
        """重置环境，返回初始状态"""
        self.current_state = self.start
        return self.current_state
    
    def step(self, action):
        """根据给定的动作，返回下一个状态，奖励，是否终止"""
        x, y = self.current_state
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # right
            y = min(self.grid_size[1] - 1, y + 1)
        elif action == 2:  # down
            x = min(self.grid_size[0] - 1, x + 1)
        elif action == 3:  # left
            y = max(0, y - 1)
        
        self.current_state = (x, y)
        
        # 检查是否到达目标
        if self.current_state == self.goal:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        
        return self.current_state, reward, done

    def get_possible_actions(self):
        """返回可能的动作集合"""
        return [0, 1, 2, 3]  # up, right, down, left


class MonteCarloPolicyIteration:
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        self.env = env
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-greedy 策略中的 epsilon 值
        self.policy = self._initialize_random_policy()  # 初始化随机策略
        self.value_function = defaultdict(float)  # 初始化状态值函数

    def _initialize_random_policy(self):
        """初始化随机策略"""
        policy = {}
        for x in range(self.env.grid_size[0]):
            for y in range(self.env.grid_size[1]):
                policy[(x, y)] = random.choice(self.env.get_possible_actions())
        return policy

    def generate_episode(self):
        """根据当前策略生成一个 episode"""
        episode = []
        state = self.env.reset()
        done = False
        while not done:
            action = self._select_action(state)
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def _select_action(self, state):
        """根据 epsilon-greedy 策略选择动作"""
        # XXX 这里很关键
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.get_possible_actions())  # 探索
        else:
            return self.policy[state]  # 利用

    def evaluate_policy(self, episodes=100):
        """通过生成多个 episodes 来评估当前策略"""
        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)
        for _ in range(episodes):
            episode = self.generate_episode()
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                if not any([state == x[0] for x in episode[:t]]):
                    returns_sum[state] += G
                    returns_count[state] += 1
                    self.value_function[state] = returns_sum[state] / returns_count[state]

    def improve_policy(self):
        """通过状态值函数来改进策略"""
        policy_stable = True
        for state in self.policy.keys():
            old_action = self.policy[state]
            # 计算每个动作的 Q 值
            action_values = np.zeros(len(self.env.get_possible_actions()))
            for action in self.env.get_possible_actions():
                next_state, reward, _ = self.env.step(action)
                action_values[action] += reward + self.gamma * self.value_function[next_state]
                self.env.current_state = state  # 还原当前状态
                
            new_action = np.argmax(action_values)
            if old_action != new_action:
                policy_stable = False
            self.policy[state] = new_action
        return policy_stable

    def policy_iteration(self, max_iterations=100):
        """蒙特卡洛策略迭代的主循环"""
        for i in range(max_iterations):
            print(f"Iteration: {i+1}")
            self.evaluate_policy()
            policy_stable = self.improve_policy()
            if policy_stable:
                print("Policy has stabilized.")
                break


# 创建 GridWorld 环境
env = GridWorldEnv()

# 创建蒙特卡洛策略迭代实例
mcpi = MonteCarloPolicyIteration(env)
mcpi.policy_iteration()

# 输出最终策略和状态值函数
print("Final policy:")
for key in sorted(mcpi.policy.keys()):
    print(f"State {key}: Action {mcpi.policy[key]}")

print("Value function:")
for key in sorted(mcpi.value_function.keys()):
    print(f"State {key}: Value {mcpi.value_function[key]:.2f}")
