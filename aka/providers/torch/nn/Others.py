import torch
import torch.nn as nn
from .Bases import Parameter
from .Containers import Functional

from torch.nn import Linear
from torch.nn import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d, GroupNorm
from torch.nn import LayerNorm
from torch.nn import AvgPool1d, AvgPool2d, AvgPool3d, MaxPool1d, MaxPool2d, MaxPool3d
from torch.nn import Dropout, Dropout1d, Dropout2d, Dropout3d
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import Embedding

class Module(nn.Module):
    def __init__(self, callback_module):
        super(Module, self).__init__()
        self.callback_module = callback_module

        i = 0
        for k in dir(callback_module):
            attr = getattr(callback_module,k)
            if(isinstance(attr,nn.Module)):
                self.add_module("L:"+str(i),attr)
                i+=1

    def forward(self, *args, **kwargs):
        return self.callback_module(*args, **kwargs)

# Tensor
def TrainModule(module, criterion):
    class TrainModuleCallback:
        def __init__(self, module, criterion):
            self.module = module
            self.criterion = criterion

        def __call__(self, inputs, targets=None):
            outputs = self.module(inputs)
            if(targets != None) :
                return outputs, self.criterion(outputs, targets) 
            else:
                return outputs
    return Module(TrainModuleCallback(module, criterion))
