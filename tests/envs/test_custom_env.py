use_rlearn = True
if use_rlearn:
    from rlearn import Env
    from rlearn.spaces import Dict, Discrete, Box
else:
    from gymnasium import Env
    from gymnasium.spaces import Dict, Discrete, Box

import numpy as np

# 定义自定义环境
class CustomEnv(Env):
    def __init__(self, *args, **kwrags):
        super(CustomEnv, self).__init__(*args, **kwrags)

        # 定义 observation_space 为 Dict
        self.observation_space = Dict({
            "position": Box(low=-1.0, high=1.0, shape=(2,)),  # 2维位置
            "velocity": Box(low=-1.0, high=1.0, shape=(2,)),  # 2维速度
            "info": Discrete(3)  # 离散信息
        })

        # 定义 action_space 为 Discrete
        self.action_space = Discrete(3)

    def reset(self):
        # 返回初始观察值
        return {
            "position": np.random.uniform(-1.0, 1.0, size=(2,)),
            "velocity": np.random.uniform(-1.0, 1.0, size=(2,)),
            "info": np.random.randint(0, 3)
        }

    def step(self, action):
        # 随机生成下一步的观察值
        observation = {
            "position": np.random.uniform(-1.0, 1.0, size=(2,)),
            "velocity": np.random.uniform(-1.0, 1.0, size=(2,)),
            "info": np.random.randint(0, 3)
        }

        # 设置奖励，假设为随机值
        reward = np.random.random()

        # 是否终止
        done = np.random.rand() > 0.95

        # 额外信息
        info = {}

        return observation, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def test_basic():
    # 创建并测试环境
    env = CustomEnv()

    obs = env.reset()
    print("Initial Observation:", obs)

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")

        if done:
            print("Episode finished!")
            break

    env.close()

if __name__ == '__main__':
    if 1:
        test_basic()
