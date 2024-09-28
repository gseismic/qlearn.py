from .reward import RewardMonitor

def get_monitor(monitor_type='reward', **kwargs):
    if monitor_type == 'reward':
        return RewardMonitor(**kwargs)
    else:
        raise ValueError(f"Unknown monitor type: {monitor_type}")