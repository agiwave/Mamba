import aka.numpy

def exist(*args, **kwargs):
    return join(*args, _raise_exceptions_for_missing_entries=False, **kwargs) is not None

def join(*args, _raise_exceptions_for_missing_entries=True, **kwargs):
    import transformers
    return transformers.utils.cached_file(*args, _raise_exceptions_for_missing_entries=_raise_exceptions_for_missing_entries, **kwargs)

def fopen(path, fname=None, ftype='file', open_kwargs={}, **kwargs):
    match ftype:
        case 'json':
            import json
            return json.load(open(join(path, fname, **kwargs), **open_kwargs))

        case 'safetensor':
            import safetensors
            return safetensors.safe_open(join(path, fname, **kwargs), framework=aka.numpy.framework(), **open_kwargs)
            
        case 'file':
            return open(join(path, fname, **kwargs), **open_kwargs)

        case 'dataset':
            import datasets
            if fname is not None:
                join(path, fname, **kwargs)
            return datasets.load_from_disk(path, **open_kwargs)

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
