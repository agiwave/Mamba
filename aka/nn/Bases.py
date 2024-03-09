from .. import boot

# Basic
def Parameter(data=None, shape=None, requires_grad=True, initializer="zeros", **kwargs): return boot.invoke()
def Input(shape): return boot.invoke()

boot.inject()