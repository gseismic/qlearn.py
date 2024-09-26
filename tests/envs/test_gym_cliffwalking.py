import gymnasium as gym
from gymnasium.wrappers import RecordVideo


def test_gym_cliffwalking():
    """
    Reward Model:
    掉落悬崖并不会结束，而是会回到起点 | Fall into the cliff will not end the episode, but will return to the starting point
    Reward:
        -1: 普通移动 | normal
        -100: 掉落悬崖 | failed
        -1: 到达目标，回合结束 | success
    Reference:
        https://gymnasium.farama.org/environments/toy_text/cliff_walking/
    """
    base_env = gym.make('CliffWalking-v0', render_mode='rgb_array')
    frame_skip = 20
    env = RecordVideo(base_env, 
                      video_folder="./videos/cliffwalking", 
                      episode_trigger=lambda e: True,
                      # video_length=1,
                      # step_trigger=lambda s: s % frame_skip == 0
                    )
    # env.metadata['render_fps'] = 1
    state, _ = env.reset()
    env.render()

    for _ in range(100):  # 假设执行100步
        action = env.action_space.sample()  # 随机动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        print(f'{action=}, {next_state=}, {reward=}, {terminated=}, {truncated=}')
        if terminated or truncated:
            env.reset()
            break

    env.close()
    
if __name__ == '__main__':
    if 1:
        test_gym_cliffwalking()
