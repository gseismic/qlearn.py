from .base import BaseAgent 

class NetAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)
    
    def to(self, device):
        '''将模型移动到指定设备 | Move model to specified device'''
        pass
    
    def cpu(self):
        '''将模型移动到CPU | Move model to CPU'''
        pass
    
    def cuda(self):
        '''将模型移动到GPU | Move model to GPU'''
        pass
    
    def eval(self):
        '''设置模型为评估模式 | Set model to evaluation mode'''
        pass
    
    def train(self):
        '''设置模型为训练模式 | Set model to training mode'''
        pass