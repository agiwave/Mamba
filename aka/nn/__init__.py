from .Bases import Input, Parameter
from .Engine import train, Trainer, load_weights, save_weights
from .Containers import *
from .Operators import *
from .Shapes import *
from .Activations import *
from .Others import *

class Object():
    def __init__(self, **kwargs): 
        for key, value in kwargs.items(): 
            setattr(self, key, value)

