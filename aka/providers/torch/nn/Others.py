import torch
import torch.nn as nn
import aka.numpy as np
from .Bases import Parameter
from .Containers import Module

from torch.nn import Linear
from torch.nn import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d, GroupNorm
from torch.nn import LayerNorm
from torch.nn import AvgPool1d, AvgPool2d, AvgPool3d, MaxPool1d, MaxPool2d, MaxPool3d
from torch.nn import Dropout, Dropout1d, Dropout2d, Dropout3d
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import Embedding

def RMSNorm(dim: int, eps: float = 1e-5):
    '''
    Reference: LLaMA and Gemma
    '''
    def forward(self, x):
        x = (x.float() * np.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return x * self.weight
    return Module(
        forward = forward,
        eps = eps,
        weight = Parameter(np.ones(dim)))
