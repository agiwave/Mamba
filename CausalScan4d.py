try:
    import torch.cuda as cuda
    import torch.utils.cpp_extension as ext
    import os
    script_dir = os.path.dirname(__file__)
    causal_scan_kernel = ext.load('extCausalScan4d', [
        os.path.join(script_dir, 'CausalScan4d.' + ('cu' if cuda.is_available() else 'cpp'))
    ])
except ImportError:
    causal_scan_kernel = None
    print('Warn: CausalScan4d import failed.')

import torch
class CausalScan(torch.autograd.Function):
    '''
    Formula:
    h(1) = a(1) * z         + b(1) * x(1)
    h(2) = a(2) * h(1)      + b(2) * x(2)
    ...
    h(n) = a(n) * h(n-1)    + b(n) * x(n)
    
    y(1) = c(1) * h(1)
    ...
    y(n) = c(n) * h(n)

    Return:
    (Y, h(n))

    Args:
    x : (b, l, d)
    h : (b, 1, d, n)
    A : (b, l, d, n)
    B : (b, l, d, n)
    C : (b, l, d, n)

    Output:
    y : (b, l, d)
    h : (b, 1, d, n)
    '''
    @staticmethod
    def forward(ctx, x, h, A, B, C):
        (x, h, A, B, C) = [item.contiguous() for item in [x, h, A, B, C]]
        x = x.unsqueeze(-1)
        for item in [x, h, A, B, C]:
            assert len(item.shape) == 4
            assert h.size(0) % item.size(0) == 0
            assert item.size(1) == 1 or item.size(1) == x.size(1)
            assert h.size(2) % item.size(2) == 0
            assert item.size(3) == 1 or item.size(3) == h.size(3)
        assert h.size(1) == 1, 'hidden_state size should be one'

        group_size = 1023   # Must match the size in kernel.
        S = torch.empty(h.size(0), (x.size(1)+group_size-1)//group_size, h.size(2), h.size(3), dtype=h.dtype, device=h.device)
        y = causal_scan_kernel.forward(x, h, S, A, B, C)
        ctx.save_for_backward(x, S, A, B, C)
        return y.squeeze(-1), h

    @staticmethod
    def backward(ctx, gradO, gradH):
        x, S, A, B, C = ctx.saved_variables
        gradX, gradH, gradA, gradB, gradC = causal_scan_kernel.backward(gradO.unsqueeze(-1), gradH, x, S, A, B, C)
        return gradX.squeeze(-1), gradH, gradA, gradB, gradC

causalScan = None if causal_scan_kernel is None else CausalScan.apply