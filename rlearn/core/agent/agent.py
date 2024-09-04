from ...logger import user_logger


class Agent:

    def __init__(self, name, env, verbose=1, logger=None):
        self.name = name
        self.env = env
        self.verbose = verbose
        self.logger = logger or user_logger
