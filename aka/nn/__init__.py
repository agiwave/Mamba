from .Bases import Input, Parameter
from .Engine import train, Trainer, load_weights, save_weights
from .Containers import *
from .Operators import *
from .Shapes import *
from .Activations import *
from .Others import *

class Args():
    def __init__(self, **kwargs): 
        self.keys = [key for key in kwargs]
        for key in kwargs: 
            setattr(self, key, kwargs[key])
    def cat(self, v):
        args = {}
        for key in self.keys:
            args[key] = getattr(self, key)
        for key in v.keys:
            args[key] = getattr(v, key)
        return Args(**args)

