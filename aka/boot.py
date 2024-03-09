import os
import sys
import inspect
import importlib

def invoke():
    '''
    Invoke placeholder.
    '''
    f_back = sys._getframe().f_back
    f_module = inspect.getmodule(f_back)
    func_name = f_module.__name__+"."+f_back.f_code.co_name
    raise TypeError("Func:"+func_name+" has not been implementted by provider.")

inject_provider_names = {}
def inject():
    '''
    Inject all functions in calling module by provider implementation.
    '''
    f_back = sys._getframe().f_back
    f_module = inspect.getmodule(f_back)
    root_name = f_module.__name__.split('.')[0]

    # -- Find provider name --
    inject_provider_name = inject_provider_names.get(root_name, None)
    if(inject_provider_name is None):
        inject_provider_name = os.environ.get(root_name+"_provider_name")
        if(inject_provider_name is None):
            print("Waining: Can't find provider name ("+ root_name + "_provider_name) in os env, use torch as default.")
            inject_provider_name = "aka.providers.torch"
        inject_provider_names[root_name] =inject_provider_name
            
    # -- Load inject module --
    inject_module_name = f_module.__name__.replace(root_name+".", inject_provider_name+".")
    inject_module = importlib.import_module(inject_module_name)    

    # -- Inject attr if exist --
    for name in dir(f_module):
        if name.startswith("_")==False:
            attr = getattr(f_module, name)
            # inject variables be defined in module(not import)
            # if (inspect.getmodule(attr)==f_module and hasattr(inject_module, name) ):
            if( hasattr(inject_module, name) ):
                # if(inspect.isfunction(attr) or inspect.isclass(attr)):
                setattr(f_module, name, getattr(inject_module,name))
            


