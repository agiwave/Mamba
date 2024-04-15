import math
import aka.nn as nn
import aka.numpy as np

def MetaLayer(**kwargs):
    '''
    Build resident meta layer by name. Include: GQA(Group-Query Attention), MLP, GateMLP, ...
    '''
    def __init__(self, **kwargs):
        layer = kwargs.get('layer', None)
        if layer is None:
            name = kwargs['name']
            import importlib
            module = importlib.import_module(name)
            short_name = name.split('./\\')[-1]
            m = getattr(module, short_name+"Block", None)
            assert m is not None, f"Unknown layer:{name}"
            self.layer = m(**kwargs)
        else:
            self.layer = layer
        self.norm = nn.RMSNorm(kwargs['latent_dim'])
        self.x_gate = None if not kwargs.get('x_gate',False) else nn.Parameter(np.ones(kwargs['latent_dim']))
        self.resident_gate = None if not kwargs.get('resident_gate',False) else nn.Parameter(np.ones(kwargs['latent_dim']))
        return self

    def forward(self, x, **kwargs):
        y = self.layer(self.norm(x), **kwargs)
        y = y if self.x_gate is None else y * np.gelu(self.x_gate)
        x = x if self.resident_gate is None else x * np.gelu(self.resident_gate)
        return x + y, None
    return __init__(nn.Module(forward = forward), **kwargs)

def ScaleDk():
    def forward(self, x):
        return x * (x.size(-1)**0.5)
    return nn.Module(forward = forward)

def CausalLM(**kwargs):
    '''
    Causal Language Model.
    '''
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        self.tokenizer = args.tokenizer
        self.latent_dim = args.latent_dim
        self.vocab_dim = getattr(args, 'vocab_dim', args.latent_dim)
        self.vocab_mode = getattr(args, 'vocab_mode', None)
        if self.vocab_dim != self.latent_dim:
            match self.vocab_mode:
                case 'onehot':
                    self.in_proj = None
                case _:
                    self.in_proj = nn.Linear(self.vocab_dim, self.latent_dim, bias=args.bias)
                    self.out_proj = nn.Linear(self.latent_dim, self.vocab_dim, bias=args.bias)

        self.embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=self.vocab_dim)
        match getattr(args, 'prev_norm', None):
            case 'rms':
                self.prev_norm = nn.RMSNorm(args.latent_dim)
            case 'scaledk':
                self.prev_norm = ScaleDk()
            case _:
                self.prev_norm = None
        self.layers = nn.ModuleList([MetaLayer(**dict(kwargs, **layer)) for layer in args.layers])
        self.lm_head = None if not getattr(args, 'lm_head', False) else nn.Linear(self.vocab_dim, args.vocab_size,bias=False)
        self.post_norm = nn.RMSNorm(self.vocab_dim)
        self.train_mode = getattr(args, 'train_mode', None)
        self.cache = {}
        return self

    def forward(self, inputs, targets=None, state=None):
        # -- Embedding and layers
        x = self.embedding(inputs)

        # -- vocab_dim --> latent_dim
        if self.vocab_dim != self.latent_dim:
            match self.vocab_mode:
                case 'onehot':
                    n = self.latent_dim // self.vocab_dim
                    h = np.arange(n, dtype=x.dtype, device=x.device).unsqueeze(-1)
                    x = x.unsqueeze(-2)
                    x = np.exp(-((x * (n-1) - h)**2))
                    x = np.rearrange('b l n d -> b l (n d)', x)
                case _:
                    x = self.in_proj(x)

        # -- layers --
        if self.prev_norm is not None:
            x = self.prev_norm(x)
            
        if(state is not None):
            layer_states = state.get('layer_states', None)
            if layer_states is None:
                layer_states = [{} for _ in self.layers]
                state['layer_states'] = layer_states

        layer_losses = []
        for i, layer in enumerate(self.layers):
            l_state = None if state is None else layer_states[i]
            x, loss = layer(x, cache=self.cache, state=l_state)
            if loss is not None:
                layer_losses.append(loss)

        # -- latent_dim --> vocab_dim
        if self.vocab_dim != self.latent_dim:
            match self.vocab_mode:
                case 'onehot':
                    # n = self.latent_dim // self.vocab_dim
                    # h = np.arange(n, dtype=x.dtype, device=x.device).unsqueeze(-1)
                    x = np.rearrange('b l (n d)->b l n d', x, n=n)
                    x = np.softmax(x, dim=-2)
                    x = np.einsum('b l n d->b l d', x*h) / (n-1)
                case _:
                    x = self.out_proj(x)

        if self.post_norm is not None:
            x = self.post_norm(x)

        # -- vocab_dim --> logits
        if self.lm_head is not None:
            y = self.lm_head(x)    # -- LLaMA vs embedding.weight ? --
        else:
            y = np.einsum('bld,nd->bln', x, self.embedding.weight) * (self.vocab_dim**-0.5)

        # -- logits --> output
        if(targets is not None):
            if self.train_mode is None:
                loss = np.cross_entropy(y.view(-1, y.size(-1)), targets.reshape(-1), ignore_index=-1)
                if len(layer_losses) > 0:
                    loss += np.sum(np.stack(layer_losses, dim=-1)) / len(layer_losses)
            else:
                assert False
            return y, loss
        else:
            return y

    def generator(self, prompts: str, max_length : int = 64):
        prompt_tokens = [self.tokenizer.bos_token_id]+self.tokenizer.encode(prompts) # [self.tokenizer.bos_token_id]+
        if hasattr(self, 'eval'):
            self.eval()

        with np.no_grad():
            state = {}
            cache = []
            input_token_ids = np.array([prompt_tokens])
            for _ in range(max_length):
                outputs = self(input_token_ids, state=state)
                input_token_ids = np.argmax(outputs[:,-1:,:], dim=-1)
                cache = cache + input_token_ids[0].tolist()
                if self.tokenizer.eos_token_id in input_token_ids:
                    break

                word = self.tokenizer.decode(cache)
                word_token_ids = self.tokenizer.encode(word)
                if len(word_token_ids)>0 and cache[-1] == word_token_ids[-1]:
                    cache = []
                    yield word

            if len(cache)>0:
                yield self.tokenizer.decode(cache)

    def generate(self, prompts : str, max_length : int = 64):
        response = ''
        for w in self.generator(prompts,max_length):
            response += w
        return response
        
    return __init__(nn.Module(forward = forward, generate = generate, generator=generator),**kwargs)

def CausalLMArgs(name):
    mlp_args = dict(
        name = 'Xproj',
        hidden_dim = 384*4,
    )
    attn_args = dict(
        name = 'Attention',
        k_dim = 384,
        hidden_dim = 384,
        num_heads = 6,
        num_kv_groups = 6,
        rotary_embedding = True,
    )
    return dict(
        vocab_size = 50304,
        vocab_dim = 64,
        block_size = 256,
        latent_dim = 384,

        dropout = 0.2,
        bias = False, # do we use bias inside LayerNorm and Linear layers?
        layers = [attn_args, mlp_args]*6,
    )

if __name__ == "__main__":
    from RomeArena import TrainRoles, RunRoles
    TrainRoles([
        'CausalLM-demo'
    ], lr = 6e-4, epochs=3)
    # RunRoles([
    #     'CausalLM-demo'
    # ], "Paul Daniels (born 4 June 1981 in Burlington)")


# One by One
# for i in range(len(prompt_tokens)-1):
#     self(np.array([prompt_tokens[i:i+1]]), state=state)
# input_token_ids = np.array([prompt_tokens[-1:]])

# Without state
# if len(prompt_tokens) > 1:
#     self(np.array([prompt_tokens[:-1]]))
# input_token_ids = np.array([prompt_tokens])
# for _ in range(max_length):
#     outputs = self(input_token_ids)
#     output_token_ids = np.argmax(outputs[:,-1:,:], dim=-1)
#     cache = cache + output_token_ids[0].tolist()
#     if self.tokenizer.eos_token_id in input_token_ids:
#         break

#     word = self.tokenizer.decode(cache)
#     word_token_ids = self.tokenizer.encode(word)
#     if cache == word_token_ids:
#         cache = []
#         yield word

#     input_token_ids = np.cat([input_token_ids, output_token_ids], dim=1)
