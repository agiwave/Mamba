import aka.nn as nn
import aka.numpy as np

def MambaBlock(**kwargs):
    """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        self.hidden_dim = getattr(args, 'hidden_dim', args.latent_dim)
        self.num_heads = getattr(args, 'num_heads', self.hidden_dim)
        self.num_states = getattr(args, 'num_states', 16)
        self.dt_rank = getattr(args, 'dt_rank', args.latent_dim//16)
        self.conv_kernel_size = getattr(args, 'conv_kernel_size', 4)

        # A = np.repeat(np.arange(1, self.num_states + 1).unsqueeze(0), self.num_heads, 0)
        A = np.repeat(np.arange(1, self.num_heads + 1).unsqueeze(1), self.num_states, 1)
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
        self.x_proj = nn.Linear(self.hidden_dim, self.dt_rank + self.num_states * 2, bias=False)
        # dt_proj projects Δ from dt_rank to hidden_dim
        self.dt_proj = nn.Linear(self.dt_rank, self.num_heads, bias=True)
        self.A_log = nn.Parameter(np.log(A))
        self.D = nn.Parameter(np.ones(self.hidden_dim))
        self.out_proj = nn.Linear(self.hidden_dim, args.latent_dim, bias=args.bias)
        return self
        
    def forward(self, x, state=None, **kwargs):
        (b, l, d) = x.shape
        (x, gate) = self.in_proj(x).chunk(2, dim=-1)
        
        # -- Load State --
        x = np.einsum('bld->bdl',x)
        if state is not None:
            n_conv_state = self.conv_kernel_size-1
            if 'conv_state' in state:
                conv_state = state['conv_state']
                ssm_state = state['ssm_state']
            else:
                conv_state = np.zeros(b, self.hidden_dim, n_conv_state, device=x.device)
                ssm_state = np.zeros(b, self.num_heads, self.hidden_dim//self.num_heads, self.num_states, device=x.device)
            x = np.cat((conv_state, x), dim=2)
        else:
            n_conv_state = 0
            ssm_state = np.zeros(b, self.num_heads, self.hidden_dim//self.num_heads, self.num_states, device=x.device)

        # -- Conv --
        if x.size(2) < l + n_conv_state:
            x = np.pad(x, (l + n_conv_state - x.size(2), 0), value=0.)
        y = self.conv1d(x)
        y = np.einsum('bdl->bld', y)
        y = np.silu(y)

        '''
        This is the classic state space formula:
            h(t + 1) = Ah(t) + Bx(t)    --- (1)
            y(t)     = Ch(t) + Dx(t)
        Formula (1) is a topic rnn.
            h(n)     = a(n) * h(n-1) + b(n)
        '''
        A = -np.exp(self.A_log.float())  # shape (hidden_dim, n)
        D = self.D.float() * y
        x_dbl = self.x_proj(y)  # (b, l, dt_rank + 2*num_states)
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, self.num_states, self.num_states], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = np.softplus(self.dt_proj(delta))  # (b, l, hidden_dim)
        # y, ssm_state = ssm(y, delta, A, B, C, D, self.num_heads, ssm_state)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        deltaA = np.einsum('blh,hn->blhn', delta, A).unsqueeze(-2)                # -> [B, L, h, n]
        y = np.rearrange('b l (h d)->b l h d', y, h=self.num_heads)
        deltaB = np.einsum('blh,bln,blhd->blhdn', delta, B, y)
        
        # -- RNN --
        cumA = np.exp(np.cumsum(deltaA, dim=1))
        mask = np.tril(np.ones(l, l, device=x.device))
        shiftA = np.pad(cumA, (0, 0, 0, 0, 0, 0, 1, -1), value=1.0)
        shiftB = np.cat([ssm_state.unsqueeze(1), deltaB[:,:l-1]], dim=1) / (1e-10+shiftA)
        S = np.einsum('blhdn,lm,bmhdn->blhdn', cumA, mask, shiftB) + deltaB
        # -- RNN --
        
        ssm_state = S[:,-1]
        y = np.einsum('blhdn,bln->blhd', S, C)
        y = np.rearrange('b l h d-> b l (h d)', y, h=self.num_heads)
        y = y + D

        # -- Save State --
        if state is not None:
            state['ssm_state'] = ssm_state.detach()
            state['conv_state'] = x[:, :, -n_conv_state:].detach()
            
        y = y * np.silu(gate)
        return self.out_proj(y)
    return __init__(nn.Module(forward = forward), **kwargs)
            
def Mamba(name):
    import aka.repo as repo
                
    # -- Tokenizer --
    tokenizer = repo.AutoTokenizer(name)
    cfg = repo.fopen(name, 'config.json', ftype='json')
    args = dict(
        tokenizer = tokenizer,
        vocab_size = cfg['vocab_size'],
        latent_dim = cfg['d_model'],
        layers = [
            dict(
                name = 'Mamba',
                hidden_dim = cfg['intermediate_size'],
                num_heads = cfg['intermediate_size'],
                dt_rank = cfg['d_model']//16,
                conv_kernel_size = 4,
                conv_bias = True,
                num_states = cfg['state_size']
            )
        ]*cfg['n_layer'],
        bias = False
    )

    # -- Model --
    from CausalLM import CausalLM
    mamba = CausalLM(**args)
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
