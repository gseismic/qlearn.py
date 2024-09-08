#import gymnasium as gym


class EnvWrapper(Env):

    def __init__(self, env, reset_return_info=True, step_return_truncated=True):
        self.env = env
        self.reset_return_info = reset_return_info
        self.step_return_truncated = step_return_truncated

    def reset(self, *args, **kwargs):
        res = self.env.reset(*args, **kwargs)
        if self.reset_return_info:
            return res
        else:
            return res[0]

    def step(self, *args, **kwargs):
        res = self.env.step(*args, **kwargs)
        if self.step_return_truncated:
            return res
        else:
            done = res[2] or res[3]
            return res[0], res[1], done, res[5]


def gym_env_wrapper(env, reset_return_info=True, step_return_truncated=True):
    return EnvWrapper(env, 
                      reset_return_info=reset_return_info, 
                      step_return_truncated=step_return_truncated)
