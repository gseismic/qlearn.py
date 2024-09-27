import numpy as np
import random
import torch

def seed_torch(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available() and torch.backends.cudnn.enabled:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def seed_all(seed):
    if seed is None:
        return
    
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
