from .. import boot

# Flow
def Module(**kwargs): return boot.invoke()
def ModuleList(modules=None): return boot.invoke()
def Sequential(*sequential_modules): return boot.invoke()
def Parrallel(*parrallel_modules, join_module=None): return boot.invoke()
def Functional(func, *args, class_method=False, **kwargs): return boot.invoke()

# Mode
def TrainModule(net, los=None):return net

boot.inject()