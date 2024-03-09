import torch
import torch.nn as nn

from torch.nn import Flatten

# Reshape模块
class Reshape(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Reshape, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.reshape(-1, *self.args)

# Transpose模块
class Transpose(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Transpose, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return torch.transpose(x, *self.args).contiguous()

# Permute
class Permute(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Permute, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.permute(0, *self.args).contiguous()

def ImageAxisC():return 0
def ImageAxisH():return 1
def ImageAxisW():return 2
def ImageShape(c,w,h):return torch.Size([c, w, h])