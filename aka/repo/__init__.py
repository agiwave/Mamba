from .. import boot

def exist(repo, pathname, **kwargs): return boot.invoke()
def fopen(repo, pathname, ftype, open_kwargs=None, **kwargs): return boot.invoke()

def AutoModel(repo): return boot.invoke()
def AutoDataset(repo): return boot.invoke()
def AutoConfig(repo): return boot.invoke()
def AutoTokenizer(repo): return boot.invoke()

boot.inject()