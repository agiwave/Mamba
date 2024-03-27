import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import aka.numpy

def exist(*args, **kwargs):
    return join(*args, _raise_exceptions_for_missing_entries=False, **kwargs) is not None

def join(*args, _raise_exceptions_for_missing_entries=True, **kwargs):
    import transformers
    return transformers.utils.cached_file(*args, _raise_exceptions_for_missing_entries=_raise_exceptions_for_missing_entries, **kwargs)

def fopen(repo, pathname, ftype='file', open_kwargs={}, framework=aka.numpy.framework(), **kwargs):
    match ftype:
        case 'json':
            import json
            return json.load(open(join(repo, pathname, **kwargs), **open_kwargs))

        case 'safetensor':
            import safetensors
            return safetensors.safe_open(join(repo, pathname, **kwargs), framework=framework, **open_kwargs)
            
        case 'file':
            return open(join(repo, pathname, **kwargs), **open_kwargs)

def AutoDataset(*args, **kwargs):
    import datasets
    return datasets.load_dataset(*args, **kwargs)

def AutoModel(*args, **kwargs):
    import transformers
    return transformers.AutoModel.from_pretrained(*args, **kwargs)

def AutoTokenizer(*args, **kwargs):
    import transformers
    return transformers.AutoTokenizer.from_pretrained(*args, **kwargs)    

def AutoConfig(*args, **kwargs):
    import transformers
    return transformers.AutoConfig.from_pretrained(*args, **kwargs)  
