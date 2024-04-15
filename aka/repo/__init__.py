from .. import boot

def exist(path, pathname, **kwargs): return boot.invoke()
def fopen(path, pathname, ftype, open_kwargs=None, **kwargs): return boot.invoke()

def AutoModel(path): return boot.invoke()
def AutoDataset(path): return boot.invoke()
def AutoConfig(path): return boot.invoke()
def AutoTokenizer(path): return boot.invoke()

boot.inject()