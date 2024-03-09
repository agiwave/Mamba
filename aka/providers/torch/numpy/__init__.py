import torch
from torch import *
from torch.nn.functional import unfold, layer_norm, linear
from torch.nn.functional import mse_loss, cross_entropy
from torch.nn.functional import relu, gelu, silu, softplus, softmax
from torch.nn.functional import embedding

from torch import int8, int16, int32, int64, short, int, long
from torch import float, float16, float32, bfloat16
from torch import set_default_dtype, set_default_device

array = tensor
repeat = repeat_interleave
def iden(inputs):
    return inputs

torch.arange
def unfold(data, kernel_size, stride=1, padding=0):
    K = kernel_size
    (B, C, H, W) = data.shape
    outputs = torch.nn.functional.unfold(data, kernel_size=kernel_size, stride=stride, padding=padding)
    return outputs.reshape([B, C*K*K, H, W])


