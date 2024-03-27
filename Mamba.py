import aka.nn as nn
import aka.numpy as np

def MambaBlock(args):
    """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""

    def forward(self, x, state=None, **kwargs):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
        Returns:
            output: shape (b, l, d)
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * hidden_dim)
        (x, res) = x_and_res.chunk(2, dim=-1)
        
        x = np.einsum('bld->bdl',x)
        state_size = 0
        if state is not None:
            state_size = self.args.conv_kernel_size-1
            if 'conv_state' in state:
                conv_state = state['conv_state']
                ssm_h = state['ssm_state']
            else:
                conv_state = np.zeros(b, self.args.hidden_dim, state_size, device=x.device)
                ssm_h = np.zeros(b, self.args.hidden_dim, self.args.d_state, device=x.device)
            x = np.cat((conv_state, x), dim=2)
            y = self.conv1d(x)[:, :, state_size:state_size+l]
            y = np.einsum('bdl->bld', y)
            y = np.silu(y)
            y, ssm_h = self.ssm(y,ssm_h)
            state['ssm_state'] = ssm_h.detach()
            state['conv_state'] = x[:, :, -state_size:].detach()
        else:
            ssm_h = np.zeros(b, self.args.hidden_dim, self.args.d_state, device=x.device)
            y = self.conv1d(x)[:, :, state_size:state_size+l]
            y = np.einsum('bdl->bld', y)
            y = np.silu(y)
            y, ssm_h = self.ssm(y,ssm_h)

        y = y * np.silu(res)
        return self.out_proj(y)

    def ssm(self, x, ssm_h):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]
        Args:
            x: shape (b, l, hidden_dim)    (See Glossary at top for definitions of b, l, hidden_dim, n...)
        Returns:
            output: shape (b, l, hidden_dim)
        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        """
        (hidden_dim, d_state) = self.A_log.shape
        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        A = -np.exp(self.A_log.float())  # shape (hidden_dim, n)
        D = self.D.float()
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*d_state)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, d_state, d_state], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = np.softplus(self.dt_proj(delta))  # (b, l, hidden_dim)
        y = self.selective_scan(x, delta, A, B, C, D, ssm_h)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        return y

    def selective_scan(self, x, delta, A, B, C, D, ssm_h):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            h(t + 1) = Ah(t) + Bx(t)
            y(t)     = Ch(t) + Dx(t)
            ------------------------
            BX = B*X
            DX = D*X
            h(t + 1) = Ah(t) + BX[t]
            y(t)     = Ch(t)
            Y = Y + DX
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            x: shape (b, l, hidden_dim)    (See Glossary at top for definitions of b, l, hidden_dim, n...)
            delta: shape (b, l, hidden_dim)
            A: shape (hidden_dim, d_state)
            B: shape (b, l, d_state)
            C: shape (b, l, d_state)
            D: shape (hidden_dim)
    
        Returns:
            output: shape (b, l, hidden_dim)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
        """
        (b, l, hidden_dim) = x.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        deltaA = np.exp(np.einsum('bld,dn->bldn', delta, A))
        deltaBX = np.einsum('bld,bln,bld->bldn', delta, B, x)
        y = np.empty((b, l, hidden_dim), device=deltaA.device)

        if False:
            for i in range(l):
                ssm_h = deltaA[:, i] * ssm_h + deltaBX[:, i]
                y[:,i] = np.einsum('bdn,bn->bd', ssm_h, C[:, i]) # BUG? the h is not h(t), it is already set to h(t+1) in prev line
        else:
            # Parallel Version without CUDA, Warning: This ver will take O(b,l,l,d,n) Memories.
            # 
            # S(n) = a(1)*...a(n)*S(0) + a(2)*...*a(n)*bx(1) + a(3)*...*a(n)*bx(2) +...+ a(n)*bx(n-1) + b(n)
            #
            upA = deltaA.unsqueeze(2)
            mask = np.tril(np.ones(l,l))
            mask = mask[:,:,None,None].unsqueeze(0)
            upA = np.where(mask==0, 1, upA)
            upA = np.cumprod(upA, dim=1)
            upA = np.where(mask==0, 0., upA)
            sB = np.cat([ssm_h.unsqueeze(1), deltaBX[:,:l-1]], dim=1)
            sumASB = np.sum(upA*sB.unsqueeze(1), dim=2)
            S = sumASB + deltaBX
            y = np.einsum('bldn,bln->bld', S, C)
            ssm_h = S[:,-1]
        return y + x * D, ssm_h

    mamba_args = args.mamba_args
    A = np.repeat(np.arange(1, mamba_args.d_state + 1).unsqueeze(0), mamba_args.hidden_dim, 0)
    return nn.Module(
        forward = forward,
        ssm = ssm,
        selective_scan = selective_scan,
        args = mamba_args,
        in_proj = nn.Linear(args.latent_dim, mamba_args.hidden_dim*2, bias=args.bias),
        conv1d = nn.Conv1d(
            in_channels=mamba_args.hidden_dim,
            out_channels=mamba_args.hidden_dim,
            bias=mamba_args.conv_bias,
            kernel_size=mamba_args.conv_kernel_size,
            groups=mamba_args.hidden_dim,
            padding=mamba_args.conv_kernel_size-1,
        ),
        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        x_proj = nn.Linear(mamba_args.hidden_dim, mamba_args.dt_rank + mamba_args.d_state * 2, bias=False),
        # dt_proj projects Δ from dt_rank to hidden_dim
        dt_proj = nn.Linear(mamba_args.dt_rank, mamba_args.hidden_dim, bias=True),
        A_log = nn.Parameter(np.log(A)),
        D = nn.Parameter(np.ones(mamba_args.hidden_dim)),
        out_proj = nn.Linear(mamba_args.hidden_dim, args.latent_dim, bias=args.bias)
    )

def Mamba(name):
    import os, json
    from transformers import AutoTokenizer
    from CausalLM import CausalLM

    tokenizer = AutoTokenizer.from_pretrained(name)
    if os.path.exists(name+"/config.json") == False:
        assert False, f"Can't find config file:{name+'/config.json'}"
        
    cfg = json.load(open(name+'/config.json'))  # Can't use AutoConfig here for ver reason.
    class Args():
        def __init__(self, **kwargs): 
            for key in kwargs:
                setattr(self, key, kwargs[key])
    args = Args(
        tokenizer = tokenizer,
        vocab_size = cfg['vocab_size'],
        latent_dim = cfg['d_model'],
        block_size = 100000,
        enable_cache = True, 
        layers = ['Mamba']*cfg['n_layer'],
        mamba_args = Args(
            hidden_dim = cfg['intermediate_size'],
            dt_rank = "auto",
            conv_kernel_size = 4,
            conv_bias = True,
            d_state = cfg['state_size']
        ),
        bias = False
    )
    args.mamba_args.dt_rank = args.latent_dim // 16

    mamba = CausalLM(args)
    if os.path.exists(name+"/model.safetensors"):
        def copy(desc, src):
            if not (desc.shape == src.shape):
                print("Unmatch shape was found.", desc.shape, src.shape)
                assert False
            desc.copy_(src)

        from safetensors import safe_open
        with safe_open(name+"/model.safetensors", framework="pt") as f:
            with np.no_grad():
                copy(mamba.embedding.weight, f.get_tensor('backbone.embeddings.weight'))
                copy(mamba.post_norm.weight, f.get_tensor(f'backbone.norm_f.weight'))
                for i in range(len(mamba.layers)):
                    copy(mamba.layers[i].norm.weight, f.get_tensor(f'backbone.layers.{i}.norm.weight'))
                    copy(mamba.layers[i].layer.A_log, f.get_tensor(f'backbone.layers.{i}.mixer.A_log'))
                    copy(mamba.layers[i].layer.D, f.get_tensor(f'backbone.layers.{i}.mixer.D'))
                    copy(mamba.layers[i].layer.conv1d.weight, f.get_tensor(f'backbone.layers.{i}.mixer.conv1d.weight'))
                    copy(mamba.layers[i].layer.conv1d.bias, f.get_tensor(f'backbone.layers.{i}.mixer.conv1d.bias'))
                    copy(mamba.layers[i].layer.dt_proj.weight, f.get_tensor(f'backbone.layers.{i}.mixer.dt_proj.weight'))
                    copy(mamba.layers[i].layer.dt_proj.bias, f.get_tensor(f'backbone.layers.{i}.mixer.dt_proj.bias'))
                    copy(mamba.layers[i].layer.in_proj.weight, f.get_tensor(f'backbone.layers.{i}.mixer.in_proj.weight'))
                    copy(mamba.layers[i].layer.out_proj.weight, f.get_tensor(f'backbone.layers.{i}.mixer.out_proj.weight'))
                    copy(mamba.layers[i].layer.x_proj.weight, f.get_tensor(f'backbone.layers.{i}.mixer.x_proj.weight'))
    return mamba

if __name__ == "__main__":
    mamba = Mamba('data/mamba-370m-hf')
    print('Model loaded')
    print(mamba.generate("Mamba is"))
