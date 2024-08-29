import random

class GridWorldEnv:
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4)):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.reset()
        
    def reset(self):
        self.current_state = self.start
        return self.current_state
    
    def step(self, action):
        # Define possible actions: 0: up, 1: right, 2: down, 3: left
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
        
        # Check if goal is reached
        if self.current_state == self.goal:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        
        return self.current_state, reward, done

    def get_possible_actions(self):
        return [0, 1, 2, 3]

def generate_episode(env, max_steps=1000, epsilon=0.1):
    episode = []
    state = env.reset()
    steps = 0

    for _ in range(max_steps):
        # 选择动作：epsilon-greedy策略
        if random.uniform(0, 1) < epsilon:
            action = random.choice(env.get_possible_actions())
        else:
            action = random.choice(env.get_possible_actions())  # 假设没有Q表，随机选择动作
        
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))
        
        # 输出调试信息
        print(f"Step: {steps}, State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}")

        if done:
            break
        
        state = next_state
        steps += 1

    #if steps == max_steps:
    #    print("Reached maximum steps, terminating episode to avoid infinite loop.")
        
    return episode


# 创建环境实例
env = GridWorldEnv()

# 生成一个episode
episode = generate_episode(env)

# 打印episode
print("Generated Episode:", episode)
