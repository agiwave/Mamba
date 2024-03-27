
import types
import torch
import torch.nn as nn
from .. import numpy as np
from torch.nn import ModuleList

def register_module_and_parameter(container, name, module):
    if( isinstance(module, torch.nn.Module)):
        container.register_module(name, module)
        return True
    else:
        if( isinstance(module, torch.Tensor) or isinstance(module, nn.Parameter)):
            container.register_parameter(name, module)
            return True
    return False

# call module
def call_module(m, *args, **kwargs) :
    if(m is None):
        return args if len(args)>1 else args[0]
    else:
        return m if callable(m) == False else m(*args, **kwargs)

# Module
class Module(nn.Module):
    def __init__(self, **kwargs):
        super(Module,self).__init__()
        for k in kwargs:
            v = kwargs[k]
            if(isinstance(v, types.FunctionType)):
                v = types.MethodType(v, self)
            setattr(self, k, v)

class Sequential(nn.Module):
    '''
    Sequential class
    '''
    def __init__(self, *args):
        super().__init__()
        self.module_list = torch.nn.ModuleList(args)

    def forward(self, x):
        for m in self.module_list:
            x = m(x)
        return x

# Functional
class Parrallel(nn.Module):
    '''
    Parrallel class
    '''
    def __init__(self, *parrallel_modules, join_module=None, **kwargs):
        super(Parrallel,self).__init__()
        
        self.join_module = join_module
        self.args = parrallel_modules
        self.kwargs = kwargs

        # register 
        if(join_module!=None):
            assert(callable(join_module))
            register_module_and_parameter(self, "join_module", join_module)
        else:
            assert len(kwargs)==0 or len(args)==0, 'One of the length in args and kwargs should be zero'

        i = 0
        for m in parrallel_modules:
            i+=1
            register_module_and_parameter(self, "P:"+str(i), m)

        for k in kwargs:
            register_module_and_parameter(self, k, kwargs[k])
    
    def forward(self, *inputs):
        args = [call_module(m, *inputs) for m in self.args]
        kwargs = dict([(k, call_module(m, *inputs)) for k, m in self.kwargs.items()])
        if(self.join_module!=None):
            return self.join_module(*args, **kwargs)
        else:
            n_args = len(args)
            if(n_args==0):
                return kwargs
            else:
                if(n_args == 1):
                    return args[0]
                else:
                    return args

# Functional
class Functional(nn.Module):
    '''
    Functional class
    '''
    def __init__(self, func, *args, **kwargs):
        super(Functional,self).__init__()
        
        l_args_keys = []
        for i in range(len(args)):
            argv = args[i]
            k = 'argv'+str(i)
            setattr(self, k, argv)
            l_args_keys.append(k)

        l_kwargs_keys = []
        for k in kwargs:
            v = kwargs[k]
            setattr(self, k, v)
            l_kwargs_keys.append(k)

        self.args_keys = l_args_keys
        self.kwargs_keys = l_kwargs_keys
        self._func_ = func
    
    def forward(self, *inputs):
        args = [getattr(self, k) for k in self.args_keys]
        kwargs = {k:getattr(self, k) for k in self.kwargs_keys}
        return self._func_(*inputs, *args, **kwargs)
