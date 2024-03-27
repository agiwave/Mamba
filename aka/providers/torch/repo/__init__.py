from aka.providers.hf.repo import *

import aka.providers.hf.repo as repo
from torchvision import datasets
def AutoDataset(repo_id, *args, cache_dir='data/torch', **kwargs):
    name = repo_id.upper()
    if hasattr(datasets, name):
        train_set = getattr(datasets, name)(*args, root=cache_dir, train=True, download=True, **kwargs)
        test_set = getattr(datasets, name)(*args, root=cache_dir, train=False, download=True, **kwargs)
        return {'train':train_set, 'test':test_set}

    return repo.AutoDataset(repo_id, *args, cache_dir=cache_dir, **kwargs)

