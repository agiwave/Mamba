import aka.nn as nn
import aka.numpy as np

def MambaBlock(args):
    """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
    def __init__(self, args):
        self.hidden_dim = getattr(args, 'hidden_dim', args.latent_dim)
        self.num_heads = getattr(args, 'num_heads', self.hidden_dim)
        self.dt_rank = getattr(args, 'dt_rank', args.latent_dim//16)
        self.conv_kernel_size = getattr(args, 'conv_kernel_size', 4)
        self.d_state = getattr(args, 'd_state', 16)

        A = np.repeat(np.arange(1, self.d_state + 1).unsqueeze(0), self.num_heads, 0)
        self.in_proj = nn.Linear(args.latent_dim, self.hidden_dim*2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            bias=getattr(args, 'conv_bias', True),
            kernel_size=self.conv_kernel_size,
            groups=self.hidden_dim,
            padding=0,
        )
        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.hidden_dim, self.dt_rank + self.d_state * 2, bias=False)
        # dt_proj projects Δ from dt_rank to hidden_dim
        self.dt_proj = nn.Linear(self.dt_rank, self.num_heads, bias=True)
        self.A_log = nn.Parameter(np.log(A))
        self.D = nn.Parameter(np.ones(self.hidden_dim))
        self.out_proj = nn.Linear(self.hidden_dim, args.latent_dim, bias=args.bias)
        return self

    def forward(self, x, state=None, **kwargs):
        (b, l, d) = x.shape
        (x, gate) = self.in_proj(x).chunk(2, dim=-1)
        
        x = np.einsum('bld->bdl',x)
        if state is not None:
            n_conv_state = self.conv_kernel_size-1
            if 'conv_state' in state:
                conv_state = state['conv_state']
                ssm_state = state['ssm_state']
            else:
                conv_state = np.zeros(b, self.hidden_dim, n_conv_state, device=x.device)
                ssm_state = np.zeros(b, self.num_heads, self.hidden_dim//self.num_heads, self.d_state, device=x.device)
            x = np.cat((conv_state, x), dim=2)
        else:
            n_conv_state = 0
            ssm_state = np.zeros(b, self.num_heads, self.hidden_dim//self.num_heads, self.d_state, device=x.device)

        if x.size(2) < l + n_conv_state:
            x = np.pad(x, (l + n_conv_state - x.size(2), 0), value=0.)
        y = self.conv1d(x)
        y = np.einsum('bdl->bld', y)
        y = np.silu(y)
        y, ssm_state = self.ssm(y,ssm_state)

        if state is not None:
            state['ssm_state'] = ssm_state.detach()
            state['conv_state'] = x[:, :, -n_conv_state:].detach()
            
        y = y * np.silu(gate)
        return self.out_proj(y)

    def ssm(self, x, ssm_state):
        """
        Args:
            x: (b, l, hidden_dim)
        Returns:
            output: shape (b, l, hidden_dim)
        """
        (hidden_dim, d_state) = self.A_log.shape
        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        A = -np.exp(self.A_log.float())  # shape (hidden_dim, n)
        D = self.D.float()
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*d_state)
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, d_state, d_state], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = np.softplus(self.dt_proj(delta))  # (b, l, hidden_dim)
        return selective_scan(x, delta, A, B, C, D, self.num_heads, ssm_state)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

    def selective_scan(x, delta, A, B, C, D, num_heads, ssm_state):
        """
        This is the classic discrete state space formula:
            h(t + 1) = Ah(t) + Bx(t)
            y(t)     = Ch(t) + Dx(t)
            except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
        Args:
            x: shape (b, l, hidden_dim)    (See Glossary at top for definitions of b, l, hidden_dim, n...)
            delta: shape (b, l, num_heads)
            A: shape (num_heads, d_state)
            B: shape (b, l, d_state)
            C: shape (b, l, d_state)
            D: shape (hidden_dim)
            ssm_state: (b, num_heads, hidden_dim//num_heads, d_state)
        Returns:
            output: shape (b, l, hidden_dim), ssm_state
        """
        (b, l, hidden_dim) = x.shape
        
        deltaA = np.exp(np.einsum('blh,hn->blhn', delta, A))
        hx = np.rearrange('b l (h d)->b l h d', x, h=num_heads)
        deltaBX = np.einsum('blh,bln,blhd->blhdn', delta, B, hx)

        # Parallel Version without CUDA, Warning: This ver will take O(b,l,l,d,n) Memories.
        # 
        # S(n) = a(1)*...a(n)*S(0) + a(2)*...*a(n)*bx(1) + a(3)*...*a(n)*bx(2) +...+ a(n)*bx(n-1) + b(n)
        #
        upA = deltaA.unsqueeze(2)                   # -> [B, L, 1, h, n]
        mask = np.tril(np.ones(l,l))
        mask = mask[:,:,None,None].unsqueeze(0)     # -> [B, L, L, 1, 1]
        upA = np.where(mask==0, 1, upA)
        upA = np.cumprod(upA, dim=1) * mask
        s_and_ab = np.cat([ssm_state.unsqueeze(1), deltaBX[:,:l-1]], dim=1) # -> [B, L, H, D, N]
        sumASB = np.einsum('blmhn,blhdn->blhdn', upA, s_and_ab)
        S = sumASB + deltaBX
        ssm_state = S[:,-1]

        S = np.rearrange('b l h d n-> b l (h d) n', S, h=num_heads)
        y = np.einsum('bldn,bln->bld', S, C)
        return y + x * D, ssm_state

    return __init__(nn.Module(forward = forward, ssm = ssm, selective_scan = selective_scan),args)
            
def Mamba(name):
    import aka.repo as repo
                
    # -- Tokenizer --
    tokenizer = repo.AutoTokenizer(name)
    cfg = repo.fopen(name, 'config.json', ftype='json')
    args = nn.Args(
        tokenizer = tokenizer,
        vocab_size = cfg['vocab_size'],
        latent_dim = cfg['d_model'],
        layers = [
            nn.Args(
                name = 'Mamba',
                hidden_dim = cfg['intermediate_size'],
                num_heads = cfg['intermediate_size'],
                dt_rank = cfg['d_model']//16,
                conv_kernel_size = 4,
                conv_bias = True,
                d_state = cfg['state_size']
            )
        ]*cfg['n_layer'],
        bias = False
    )

    # -- Model --
    from CausalLM import CausalLM
    mamba = CausalLM(args)
    if repo.exist(name, "model.safetensors"):
        with repo.fopen(name, "model.safetensors", ftype='safetensor') as f:
            with np.no_grad():
                mamba.embedding.weight.copy_(f.get_tensor('backbone.embeddings.weight'))
                mamba.post_norm.weight.copy_(f.get_tensor('backbone.norm_f.weight'))
                for i in range(len(mamba.layers)):
                    mamba.layers[i].norm.weight.copy_(f.get_tensor(f'backbone.layers.{i}.norm.weight'))
                    mamba.layers[i].layer.A_log.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.A_log'))
                    mamba.layers[i].layer.D.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.D'))
                    mamba.layers[i].layer.conv1d.weight.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.conv1d.weight'))
                    mamba.layers[i].layer.conv1d.bias.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.conv1d.bias'))
                    mamba.layers[i].layer.dt_proj.weight.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.dt_proj.weight'))
                    mamba.layers[i].layer.dt_proj.bias.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.dt_proj.bias'))
                    mamba.layers[i].layer.in_proj.weight.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.in_proj.weight'))
                    mamba.layers[i].layer.out_proj.weight.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.out_proj.weight'))
                    mamba.layers[i].layer.x_proj.weight.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.x_proj.weight'))
    return mamba

if __name__ == "__main__":
    mamba = Mamba('data/mamba-370m-hf')
    print('Model loaded')
    for w in mamba.generator("Mamba is"):
        print(w, end='')
