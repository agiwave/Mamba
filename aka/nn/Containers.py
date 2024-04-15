from .. import boot

def Module(**kwargs): return boot.invoke()
def ModuleList(modules=None): return boot.invoke()
def ModuleDict(modules=None): return boot.invoke()
def ParameterList(values=None): return boot.invoke()
def ParameterDict(values=None): return boot.invoke()
def Parrallel(*parrallel_modules, join_module=None): return boot.invoke()
def Functional(func, *args, **kwargs): return boot.invoke()

boot.inject()

def Sequential(*sequential_modules, loss_metric=None):
    def forward(self, inputs, *, targets=None, **kwargs):
        x = inputs
        for m in self.sequential_modules:
            x = m(x, **kwargs)
        if targets is None:
            return x
        else:
            return x, self.loss_metric(x, targets)

    return Module(
        forward = forward,
        sequential_modules = ModuleList(sequential_modules),
        loss_metric = loss_metric
    )
    