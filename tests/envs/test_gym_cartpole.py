import gymnasium as gym
from gymnasium.wrappers import RecordVideo

"""
终止条件:
    杆子倾斜超过15度
    小车移动超出中心2.4个单位
    回合达到500步
---
Termination conditions:
    The pole tilts more than 15 degrees
    The cart moves more than 2.4 units from the center
    The episode reaches 500 steps

Action Space
The action is a ndarray with shape (1,) which can take values {0, 1} indicating the direction of the fixed force the cart is pushed with.
    0: Push cart to the left
    1: Push cart to the right

Reference:
    - https://gymnasium.farama.org/environments/classic_control/cart_pole/
"""

def test_gym_cartpole():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = RecordVideo(env, 
                      video_folder="./videos/cartpole", 
                      episode_trigger=lambda e: True,
                    )

    state, _ = env.reset()
    for i in range(100):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        print(f'{i}: {next_state=}, {reward=}, {terminated=}, {truncated=}')
        if terminated or truncated:
            env.reset()
            break
    env.close()
    
if __name__ == '__main__':
    if 1:
        test_gym_cartpole()