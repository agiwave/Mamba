from .. import boot
from .. import numpy

# Connection
def Conv2d(*args, **kwargs):return boot.invoke()
def Conv1d(*args, **kwargs):return boot.invoke()
def ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0):return boot.invoke()
def Linear(*args, **kwargs):return boot.invoke()

# Others
def Dropout(rate=0.5):return boot.invoke()

# Normalization
def BatchNorm1d(*args, **kwargs):return boot.invoke()
def BatchNorm2d(*args, **kwargs):return boot.invoke()
def LayerNorm(normalize_shape):return boot.invoke()
def GroupNorm(num_groups, num_channels, eps=1e-05, affine=True):return boot.invoke()

# Pooling
def MaxPool1d(*args, **kwargs):return boot.invoke()
def MaxPool2d(*args, **kwargs):return boot.invoke()
def AvgPool2d(kernel_size, stride=None, padding=0):return boot.invoke()
def AvgPool1d(*args, **kwargs):return boot.invoke()

# Loss
def CrossEntropyLoss(*args, **kwargs):return boot.invoke()
def MSELoss():return boot.invoke()

# Embeddings
def Embedding(num_embeddings, embedding_dim):return boot.invoke()

boot.inject()

def Unfold(kernel_size, stride, padding=1):
    from .Containers import Functional
    return Functional(numpy.unfold, kernel_size, stride=stride, padding=padding)

