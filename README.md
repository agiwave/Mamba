# Mamba

A simple mamba demo with minimized code. This repo is just for learning and backup. I have no CUDA device yet. So, the code can only run on CPU for now.

The major diff to mamba-minimal is: integrated the cache feature to fast generate performance.

Please ignore all aka code here. It's a sample proxy to torch:

    aka.nn --> torch.nn
    aka.numpy --> torch + torch.nn.F

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
