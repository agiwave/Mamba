import torch
import torch.nn as nn

def Parameter(data=None, shape=None, initializer="zeros", requires_grad=True):
    if(data == None):
        data = torch.empty(shape)
        match initializer:
            case "zeros":
                nn.init.zeros_(data)
            case "ones":
                nn.init.ones_(data)
            case "xavier_uniform":
                nn.init.xavier_uniform_(data)
            case "xavier_normal":
                nn.init.xavier_normal_(data)
            case "kaiming_normal":
                nn.init.kaiming_normal_(data)
            case "kaiming_uniform":
                nn.init.kaiming_uniform_(data)
            case "randn":
                data = torch.randn(shape)
            case _:
                raise TypeError("Unknown data initializer:" + initializer)
                nn.init.kaiming_normal_(data)
    
    return torch.nn.Parameter(data=torch.Tensor(data), requires_grad=requires_grad)

# 输入
class Input(nn.Module):
    def __init__(self, shape):
        super(Input, self).__init__()
        self.shape = shape

    def forward(self, inputs):
        if(inputs.shape[1:] != self.shape):
            raise BaseException("Input shape error. Except: ", self.shape, ", But found :", inputs.shape[1:])
        return inputs
