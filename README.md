# Mamba

A simple mamba demo with minimized code. (Note: The code was tested on CPU only for now.)

The major diff to mamba-minimal is: 

1. integrated the cache feature to fast generate-performance.
2. support parallel training(not recurrentlly).

The diff to mamba is: 

1. Support input n-length tensor in infer mode not only one.
2. Support parallel training on CPU(without CUDA)
3. Support num_heads arg which is not element-wise only. The oringal mamba's num_heads = dim, it means element-wise. 

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
