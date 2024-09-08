import config
import numpy as np
from rlearn.envs import grid_world
import torch
import random
random.seed(36)
np.random.seed(36)
torch.manual_seed(36)

def test_env_gridworld_v2():
    env = grid_world.GridWorldEnv_V2(shape=(3, 5))
    env.reset()
    env.render()
    env.step(0) # up
    print('step 0')
    env.render()
    env.step(1) # right
    print('step 1')
    env.render()
    env.step(2) # down
    print('step 2')
    env.render()
    env.close()


if __name__ == '__main__':
    if 1:
        test_env_gridworld_v2()
