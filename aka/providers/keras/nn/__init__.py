
import aka.boot as boot
import keras.layers as layers
import keras.models as models
import keras.losses as losses
import keras.activations as activations


from .Bases import Parameter
from .Operators import Add, Mul, Div, Dot, MatMul
from .Containers import Sequential, Functional 

_name_maps_ = {
    # "Conv2d": "Conv2D",
    # "MaxPool2d": "MaxPool2D",
}

# 工厂函数
def create_object(module_name, type_name, *args, **kwargs):
    if(type_name in globals().keys()):
        return globals()[type_name](*args, **kwargs)
    
    type_name = _name_maps_.get(type_name, type_name)
    if(not hasattr(layers, type_name)):
        raise TypeError("Type not implement: "+type_name)

    return getattr(layers, type_name)(*args, **kwargs)

def aka_to_tf_padding(padding):
    if(padding==1):
        return 'same'
    else:
        return 'valid'


def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
    padding_mode='zeros', device=None, dtype=None, **kwargs): 
    return layers.Conv2D(
        out_channels,
        kernel_size,
        use_bias=bias,
        strides=stride,
        padding=aka_to_tf_padding(padding),
        # data_format=None,
        # dilation_rate=(1, 1),
        # activation=None,
        # kernel_initializer='glorot_uniform',
        # bias_initializer='zeros',
        # kernel_regularizer=None,
        # bias_regularizer=None,
        # activity_regularizer=None,
        # kernel_constraint=None,
        # bias_constraint=None,
    )

def MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False): 
    return layers.MaxPool2D(
        pool_size=kernel_size,
        strides=stride,
        padding=aka_to_tf_padding(padding),
        # data_format=None,
    )

def AvgPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False): 
    return layers.AvgPool2D(
        pool_size=kernel_size,
        strides=stride,
        padding=aka_to_tf_padding(padding),
        # data_format=None,
    )

def Linear(in_features, out_features, bias=True):
    return layers.Dense(units=out_features, use_bias=bias, activation=None)

def Tanh():
    return layers.Activation('tanh')
    
def Sequential(*args):
    return models.Sequential(args)

def CrossEntropyLoss():
    return losses.SparseCategoricalCrossentropy()

def TrainModule(module, criterion=None):
    return module

def BatchNorm2d(num_features):
    return layers.BatchNormalization()

def ImageAxisC():return 2
def ImageAxisH():return 0
def ImageAxisW():return 1
def ImageShape(c,w,h) :
    return [w, h, c]

# 求和
class Sum(layers.Layer):
    def __init__(self, *args):
        super().__init__()
        self.modules = args

    def call(self, x):
        y = None
        for m in self.modules:
            if( y == None ):
                y = m(x)
            else:
                y = y+m(x)
        return y

# 残差
class Resident(layers.Layer):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def call(self, x):
        return x+self.block(x)
