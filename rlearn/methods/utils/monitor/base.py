

class BaseMonitor:
    def __init__(self):
        pass

    def reset(self):
        pass

    def update(self, next_state, reward, terminated, truncated, info):
        pass

    def check_exit_conditions(self):
        pass

    def get_monitor_info(self):
        pass
