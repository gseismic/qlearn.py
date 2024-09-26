import gymnasium as gym
from gymnasium.wrappers import RecordVideo

"""
Continuous to Continuous
Transition Dynamics:
Given an action, the mountain car follows the following transition dynamics:

velocityt+1 = velocityt+1 + force * self.power - 0.0025 * cos(3 * positiont)
positiont+1 = positiont + velocityt+1

Reference:
    - https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/ 
"""

def test_gym_mountaincar():
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')

    env = RecordVideo(env, 
                      video_folder="./videos/MountainCarContinuous-v0", 
                      episode_trigger=lambda e: True,
                    )

    state, _ = env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        print(f'{i}: {action=}, {next_state=}, {reward=}, {terminated=}, {truncated=}')
        if terminated or truncated:
            env.reset()
            break
    env.close()
    
if __name__ == '__main__':
    if 1:
        test_gym_mountaincar()