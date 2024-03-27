from .. import boot

def framework(): return boot.invoke()
def no_grad(): return boot.invoke()

def iden(data, dtype=None): return boot.invoke()
def array(data, dtype=None): return boot.invoke()
def rand(shape, dtype=None): return boot.invoke()
def randn(shape, dtype=None): return boot.invoke()
def full(shape, fill_value, dtype=None): return boot.invoke()
def empty(shape, *, dtype=None): return boot.invoke()
def zeros(shape, *, dtype=None): return boot.invoke()
def zeros_like(input, *, dtype=None): return boot.invoke()
def ones(shape, dtype=None): return boot.invoke()
def ones_like(input, *, dtype=None): return boot.invoke()
def empty(shape, dtype=None): return boot.invoke()
def linspace(start, end, steps, *, dtype=None): return boot.invoke()

def nan_to_num(input, nan=0.0, posinf=None, neginf=None): return boot.invoke()

def polar(abs, angle): return boot.invoke()
def tril(input, diagonal=0): return boot.invoke()
def triu(input, diagonal=0): return boot.invoke()
def round(input, *, decimals=0): return boot.invoke()
def manual_seed(seed): return boot.invoke()
def ceil(input): return boot.invoke()       # round up
def floor(input): return boot.invoke()      # round down
def trunc(input): return boot.invoke()      # round to zero
def masked_fill(input, mask, value): return boot.invoke()

def add(*args): return boot.invoke()
def sub(*args): return boot.invoke()
def mul(*args): return boot.invoke()
def div(*args): return boot.invoke()
def dot(*args): return boot.invoke()
def matmul(*args): return boot.invoke()
def bmm(input, mat2): return boot.invoke()
def outer(input, vec2): return boot.invoke()

def permute(input, dims): return boot.invoke()
def reshape(input, shape): return boot.invoke()
def repeat(input, repeats, dim=None): return boot.invoke()
def flatten(input, start_dim=0, end_dim=-1): return boot.invoke()
def eye(n, m=None): return boot.invoke()
def view_as_real(input): return boot.invoke()
def view_as_complex(input): return boot.invoke()

def einsum(equation, *operands): return boot.invoke()
def rearrange(equation, *operands, **kwargs): return boot.invoke()

def cumsum(input, dim): return boot.invoke()
def cumprod(input, dim): return boot.invoke()
def sum(input): return boot.invoke()
def prod(input): return boot.invoke()
def mean(input): return boot.invoke()
def sqrt(x): return boot.invoke()
def rsqrt(x): return boot.invoke()
def sin(x): return boot.invoke()
def cos(x): return boot.invoke()
def exp(x): return boot.invoke()
def log(x): return boot.invoke()
def pow(input, exponent): return boot.invoke()
def abs(input): return boot.invoke()

def le(input, other): return boot.invoke()
def lt(input, other): return boot.invoke()

def cat(tensors, dim=0, descending=False, stable=False): return boot.invoke()
def split(tensor, split_size_or_sections, dim=0): return boot.invoke()
def sort(input, dim=-1): return boot.invoke()
def max(input): return boot.invoke()
def min(input): return boot.invoke()
def max(input, dim, keepdim=False): return boot.invoke()
def min(input, dim, keepdim=False): return boot.invoke()
def argsort(input, dim=-1, descending=False, stable=False): return boot.invoke()
def argmax(input, dim=None, keepdim=False): return boot.invoke()
def argmin(input, dim=None, keepdim=False): return boot.invoke()
def argwhere(input): return boot.invoke()

def gather(input, dim, index, *, sparse_grad=False): return boot.invoke()
def select(input, dim, index): return boot.invoke()
def index_select(input, dim, index): return boot.invoke()
def masked_select(input, mask): return boot.invoke()
def narrow(input, dim, start, length): return boot.invoke()
def arange(end, *, requires_grad=False): return boot.invoke()
def meshgrid(*tensors, indexing=None): return boot.invoke()
def stack(tensors, dim=0): return boot.invoke()
def flatten(input, start_dim=0, end_dim=-1): return boot.invoke()

def chunk(input, chunks, dim=0): return boot.invoke()
def where(condition, input, other): return boot.invoke()

# -- nn --
def unfold(data, kernel_size, stride=1, padding=0): return boot.invoke()
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05): return boot.invoke()
def linear(input, weight, bias=None): return boot.invoke()
def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False): return boot.invoke()

def relu(input): return boot.invoke()
def silu(input): return boot.invoke()
def swish(input): return boot.invoke()
def gelu(input, approximate='none'): return boot.invoke()
def softmax(input, dim=None): return boot.invoke()
def softplus(input, beta=1, threshold=20): return boot.invoke()
def sigmoid(input): return boot.invoke()
def tanh(input): return boot.invoke()

def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'): return boot.invoke()
def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0): return boot.invoke()
def topk(input, k, dim=None, largest=True, sorted=True): return boot.invoke()
def multinomial(input, num_samples, replacement=False, *, generator=None): return boot.invoke()

def device(device): return boot.invoke()
def set_default_dtype(d): return boot.invoke()
def load(f, *, weights_only=False, mmap=None):return boot.invoke()
def save(obj, f): return boot.invoke()

def roll(input, shifts, dims=None): return boot.invoke()
def pad(input, pad, mode='constant', value=None): return boot.invoke()
def vmap(func, in_dims=0, out_dims=0): return boot.invoke()

# -- dtype --
int8, int16, int32, int64, short, int, long = None, None, None, None, None, None, None
float, float16, float32, bfloat16 = None, None, None, None

boot.inject()