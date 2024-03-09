from .. import boot

# Activation
def ReLU(): return boot.invoke()
def GELU(): return boot.invoke()
def Tanh():return boot.invoke()
def Sigmoid():return boot.invoke()
def Softmax(*args, **kwargs):return boot.invoke()
def LeakyReLU(*args, **kwargs):return boot.invoke()

boot.inject()