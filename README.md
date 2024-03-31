# Mamba

A simple mamba demo with minimized code. (Note: The code was tested on CPU only for now.)

The major diff to mamba-minimal is: 

1. integrated the cache feature to fast generate-performance.
2. support parallel training(not recurrentlly).

The diff to mamba is: 

1. Support input n-length tensor in infer mode not only one.
2. Support parallel training on CPU(without CUDA), :)
3. Support num_heads arg which is not element-wise only. The oringal mamba's num_heads = dim, it means element-wise.

I'm not sure whether I'm the first one to realizing parallel-RNN without CUDA kernel? If possible, let me know if someone did it before. The formula is:

```
s(1) = a(1) * s(0) + b(1)
s(2) = a(1) * a(2) * s(0) + a(2)*b(1) + b(2)
s(n) = a(n) * s(n-1) + b(n)
     = a(1) *...* a(n) * s(0) + a(2) *...*a(n) * b(1) + .... + a(n-1) * b(n-1) + b(n)
cuma = [a(1), a(1) * a(2), ..., a(1)*...*a(n)] = np.cumprod(a)
shifta = [ 1., cuma(1), cuma(2), ...., cuma(n-1)] = 
shiftb = [ s(0), b(1), ..., b(n-1)]
s(n) = cuma(n) * ( shiftb(1) / shifta(1) + shiftb(2) / shifta(2) + .... + shiftb(n) / shifta(n)) + b(n)
```

Please ignore all aka code here. It's a sample proxy to torch:

    aka.nn --> torch.nn
    aka.numpy --> torch + torch.nn.functional
    aka.repo --> datasets

## Requirements

    python
    torch
    torchvision
    sentencepiece
    transformer

## Prepare

Download mamba files from: https://huggingface.co/state-spaces/mamba-370m-hf

to folder:

    data/mamba-370m-hf

## Run

python Mamba.py
