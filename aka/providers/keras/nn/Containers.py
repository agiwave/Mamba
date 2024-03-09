
import keras.models as models
from keras.layers import Layer as Module

def register_module_and_parameter(container, name, module):
    if( isinstance(module, layers.Layer) or isinstance(module, models.Model)):
        setattr(container, name, module)
        return True
    else:
        if( isinstance(module, tf.Variable) ):
            setattr(container, name, module)
            return True
    return False

def make_callable_op(m):
    return m if callable(m) else lambda *args : m

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
            assert(join_module is callable)
            register_module_and_parameter(self, "join_module", join_module)

        i = 0
        for m in parrallel_modules:
            i+=1
            register_module_and_parameter(self, "P:"+str(i), m)

        for k in kwargs:
            register_module_and_parameter(self, k, kwargs[k])
    
    def forward(self, *inputs):
        parrallel_results = [call_module(m, *inputs) for m in self.parrallel_modules]
        kwargs = dict([(k, call_module(m, *inputs)) for k, m in self.kwargs.items()])
        if(self.join_module!=None):
            return self.join_module(*parrallel_results, **kwargs)
        else:
            return parrallel_results

# Functional
class Functional(Module):
    '''
    Functional class
    '''
    def __init__(self, func, *args, **kwargs):
        super(Functional,self).__init__()
        
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.n_args = len(args)+len(kwargs.items())

        # register
        register_module_and_parameter(self, "Func", func)
        i = 0
        for m in args:
            i+=1
            register_module_and_parameter(self, "Arg:"+str(i), m)

        for k in kwargs:
            register_module_and_parameter(self, k, kwargs[k])
    
    def forward(self, *inputs):
        return self.func(*self.args, **self.kwargs)