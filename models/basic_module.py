import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import time
import os
class BasicModule(torch.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        
        prefix = 'check_points/' + self.model_name +name+ '/'
        if not os.path.isdir(prefix):
            os.mkdir(prefix)
        name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        print('model name', name.split('/')[-1] )
        torch.save(self.state_dict(), name)
        torch.save(self.state_dict(), prefix+'latest.pth')
        return name
    
    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    
    def load_latest(self, notes):
        path = 'check_points/' + self.model_name +notes+ '/latest.pth'
        self.load_state_dict(torch.load(path))