import numpy as np
import random
import torch

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # 如果使用 CUDA，确保所有的 GPU 设置相同的种子。
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
    # 确保所有 PyTorch 的操作都是确定性的 | Ensure PyTorch operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
