import torch
import numpy

from torch import *
from torch.nn.functional import unfold, layer_norm, linear
from torch.nn.functional import mse_loss, cross_entropy
from torch.nn.functional import relu, gelu, silu, softplus, softmax
from torch.nn.functional import embedding
from torch.nn.functional import pad
from torch.nn.functional import conv1d, conv2d, conv3d

from torch import int8, int16, int32, int64, short, int, long
from torch import float, float16, float32, bfloat16
from torch import set_default_dtype, set_default_device
from PIL import Image

def framework():
    return 'pt'

def array(data, *args, **kwargs):
    if isinstance(data, Image.Image):
        img = data
        if hasattr(img, "getbands"):
            c = len(img.getbands())
        else:
            c = img.channels
        img = tensor(numpy.array(img), *args, **kwargs)
        match img.ndim:
            case 2:
                return rearrange('w h -> c h w', img, c=c)
            case 3:
                return rearrange('w h c-> c h w', img, c=c)
            case _:
                assert False
    return tensor(data, *args, **kwargs)
    
repeat = repeat_interleave
swish = silu
def iden(inputs):
    return inputs

def unfold(data, kernel_size, stride=1, padding=0):
    K = kernel_size
    (B, C, H, W) = data.shape
    outputs = torch.nn.functional.unfold(data, kernel_size=kernel_size, stride=stride, padding=padding)
    return outputs.reshape([B, C*K*K, H, W])

def rearrange(equation, *operands, **kwargs):
    import einops
    if len(operands) == 1:
        return einops.rearrange(*operands, equation, **kwargs)
    else:
        return einops.rearrange(operands, equation, **kwargs)
