
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Layer as Module

def register_module_and_parameter(container, name, module):
    if( isinstance(module, layers.Layer) or isinstance(module, models.Model)):
        setattr(container, name, module)
        return True
    else:
        if(isinstance(module, tf.Variable)):
            setattr(container, name, module)
            return True
    return False

# call module
def call_module(m, *args, **kwargs) :
    return m(*args) if callable(m) else m

def Sequential(*args):
    return models.Sequential(args) 

# Functional
class Parrallel(Module):
    '''
    Parrallel class
    '''
    def __init__(self, *parrallel_modules, join_module=None, **kwargs):
        super(Parrallel,self).__init__()
        
        self.join_module = join_module
        self.parrallel_modules = parrallel_modules
        self.kwargs = kwargs

        # register 
        if(join_module!=None):
            assert(callable(join_module))
            register_module_and_parameter(self, "join_module", join_module)

        i = 0
        for m in parrallel_modules:
            i+=1
            register_module_and_parameter(self, "P:"+str(i), m)

        for k in kwargs:
            register_module_and_parameter(self, k, kwargs[k])
    
    def call(self, *inputs):
        parrallel_results = [call_module(m, *inputs) for m in self.parrallel_modules]
        kwargs = dict([(k, call_module(m, *inputs)) for k, m in self.kwargs.items()])
        if(self.join_module!=None):
            return self.join_module(*parrallel_results, **kwargs)
        else:
            n_args = len(parrallel_results)
            if(n_args==0):
                return kwargs
            else:
                if(n_args == 1):
                    return parrallel_results[0]
                else:
                    return parrallel_results
# Functional
# class Functional(Module):
#     '''
#     Functional class
#     '''
#     def __init__(self, func, *args, **kwargs):
#         super(Functional,self).__init__()
        
#         self.func = func
#         self.args = args
#         self.kwargs = kwargs

#         # register
#         register_module_and_parameter(self, "Func", func)
#         i = 0
#         for m in args:
#             i+=1
#             register_module_and_parameter(self, "Arg:"+str(i), m)

#         for k in kwargs:
#             register_module_and_parameter(self, k, kwargs[k])
    
#     def call(self, *inputs):
#         return self.func(*inputs, *self.args, **self.kwargs)


# Functional
class Functional(Module):
    '''
    Functional class
    '''
    def __init__(self, func, *args, class_method=False, **kwargs):
        super(Functional,self).__init__()
        
        l_args_keys = []
        for i in range(len(args)):
            argv = args[i]
            if(class_method == True and isinstance(argv, types.FunctionType)):
                argv = types.MethodType(argv, self)
            k = 'argv'+str(i)
            setattr(self, k, argv)
            l_args_keys.append(k)

        l_kwargs_keys = []
        for k in kwargs:
            v = kwargs[k]
            if(class_method == True and isinstance(v, types.FunctionType)):
                v = types.MethodType(v, self)
            setattr(self, k, v)
            l_kwargs_keys.append(k)

        self._func_ = func if(class_method==False) else types.MethodType(func,self)
        self._method_ = class_method
        self.args_keys = l_args_keys
        self.kwargs_keys = l_kwargs_keys

    def call(self, *inputs):
        if(self._method_):
            return self._func_(*inputs)
        else:
            args = [getattr(self, k) for k in self.args_keys]
            kwargs = {k:getattr(self, k) for k in self.kwargs_keys}
            return self._func_(*inputs, *args, **kwargs)