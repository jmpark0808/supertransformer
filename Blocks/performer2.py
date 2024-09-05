import torch, math
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from math import ceil
from functools import partial
from contextlib import contextmanager
from Blocks.swin_common import PatchMerging, PatchExpandLowerDim, BasicLayerUpsampleMA
from Blocks.performer_diffpool import TFMDecoder
from Blocks.performer_diffpool import PerformerEncoderToken, TransformerEncoderToken
from Blocks.performer_diffpool import TransformerDecoder as PerformerDecoder
from Blocks.TransformerBlocks import Transformer as TFM
from Blocks.diffslic_og import DiffSLIC, spixel_upsampling

from torch_geometric.utils import scatter

# from Blocks.GraphPooling import TopKPooling
def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

# def cast_tuple(val):
#     return (val,) if not isinstance(val, tuple) else val

def get_module_device(module):
    return next(module.parameters()).device

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val

# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash.type_as(data)

def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    
    b, h, *_ = data.shape
    
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon
    
    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)
    
    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)

def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    q, r = torch.qr(unstructured_block.cpu(), some = True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
#     print("linear attention", out.size)
    return out

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), no_projection = False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal
        
    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device
        
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
#         print('fastattention', out.size())
        return out

# a module for keeping track of when to update the projections

class ProjectionUpdater(nn.Module):
    def __init__(self, instance, feature_redraw_interval):
        super().__init__()
        self.instance = instance
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projections_(self):
        self.feature_redraw_interval = None

    def redraw_projections(self):
        model = self.instance

        if not self.training:
            return

        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(model)

            fast_attentions = find_modules(model, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x):
        raise NotImplemented


class SeparableLinear(nn.Module):
    def __init__(self, in_dim, out_dim, tokens, bias=True):
        super().__init__()
        self.in_features = in_dim
        self.out_features = out_dim
        self.weight = nn.Conv1d(in_dim, out_dim, 1, groups=tokens, bias=bias)
        
    def forward(self, input):
        # input = (B, N, D)
        input = input.permute(0, 2, 1) # (B, D, N)
        input = self.weight(input)
        input = input.permute(0, 2, 1) # (B, N, D)
        return input
# classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        heads = 4,
        dim_head = 32,
        local_heads = 0,
        local_window_size = 256,
        nb_features = None,
        feature_redraw_interval = 1000,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        dropout = 0.,
        no_projection = False,
        qkv_bias = False,
        attn_out_bias = True,
        tokens = 1
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads 
        
        self.fast_attention = FastAttention(dim_head, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)

        self.heads = heads
        self.tokens = tokens
        self.global_heads = (heads - local_heads)
        # self.local_attn = LocalAttention(window_size = local_window_size, causal = causal, autopad = True, dropout = dropout, look_forward = int(not causal), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None

        # self.to_q = SeparableLinear(dim, inner_dim, tokens, bias = qkv_bias)
        # self.to_k = SeparableLinear(dim, inner_dim, tokens, bias = qkv_bias)
        # self.to_v = SeparableLinear(dim, inner_dim, tokens, bias = qkv_bias)
        # self.to_out = SeparableLinear(inner_dim, dim, tokens, bias = qkv_bias)#nn.Linear(inner_dim, dim, bias = attn_out_bias)
        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        # self.to_out = nn.Linear(inner_dim, dim, bias = attn_out_bias)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, pos_emb = None, context = None, mask = None, context_mask = None, **kwargs):
        
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads
        
        cross_attend = exists(context)
        # print(x.size(), context.size())
        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask
  
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))
        
        attn_outs = []
        
        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            if exists(pos_emb) and not cross_attend:
                q, k = apply_rotary_pos_emb(q, k, pos_emb)

            out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
#         print("Attention", out.size())
        # out =  self.to_out(out)
        out = self.dropout(out)
        return out


class SelfAttention(Attention):
    def forward(self, *args, context = None, **kwargs):
        assert not exists(context), 'self attention should not receive context'
#         print(1, "self attention module")
        return super().forward(*args, **kwargs)

class CrossAttention(Attention):
    def forward(self, *args, context = None, **kwargs):
        assert exists(context), 'cross attention should receive context'
        return super().forward(*args, context = context, **kwargs)
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., attn_dropout=0., num_tokens=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        local_attn_heads = 0
        local_window_size = 256
        causal = False
        nb_features = None
        generalized_attention = True
        kernel_fn = nn.ReLU()
        # attn_dropout = 0.
        no_projection = False
        qkv_bias = True
        attn_out_bias = True
        self.num_tokens = num_tokens
        self.heads = heads
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_attn_heads,
                                            local_window_size = local_window_size, nb_features = nb_features,
                                              generalized_attention = generalized_attention, kernel_fn = kernel_fn,
                                                dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias,
                                                  attn_out_bias = attn_out_bias, tokens=num_tokens)),
                PreNorm(dim//heads, FeedForward(dim//heads, dim, heads, dropout = dropout))
            ]))
    def forward(self, x):
        # x = (B, N, D)
        B, N, _ = x.shape
        # x = x.reshape(B, N, self.num_tokens, -1).permute(0, 2, 1, 3)
        # x = x.reshape(B*self.num_tokens, N, -1)
        for attn, ff in self.layers:
            x = attn(x) + x
            x_ = rearrange(x, 'b n (h d) -> (b h) n d', h = self.heads)
            # x = rearrange(x, '(b t) n d -> b n (t d)', t = self.num_tokens)
            x = ff(x_) + x
            # x = rearrange(x, 'b h n d -> b n (h d)')

        return x
    

'''
class TransformerEncoder(nn.Module):
    def __init__(self, dim, nodes, depth, heads, dim_head, mlp_dim, dropout = 0., attn_dropout=0., downsample=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        local_attn_heads = 0
        local_window_size = 256
        causal = False
        nb_features = None
        generalized_attention = True
        kernel_fn = nn.ReLU()
        no_projection = False
        qkv_bias = True
        attn_out_bias = True
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_attn_heads,
                                            local_window_size = local_window_size, nb_features = nb_features,
                                              generalized_attention = generalized_attention, kernel_fn = kernel_fn,
                                                dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias,
                                                  attn_out_bias = attn_out_bias)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        if downsample:
            self.observer = nn.Parameter(torch.randn(1, nodes, dim))
            self.downsample = CrossAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_attn_heads,
                                            local_window_size = local_window_size, nb_features = nb_features,
                                              generalized_attention = generalized_attention, kernel_fn = kernel_fn,
                                                dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias,
                                                  attn_out_bias = attn_out_bias)
            # self.downsample = nn.Sequential(PreNorm(dim, CrossAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_attn_heads,
            #                                 local_window_size = local_window_size, nb_features = nb_features,
            #                                   generalized_attention = generalized_attention, kernel_fn = kernel_fn,
            #                                     dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias,
            #                                       attn_out_bias = attn_out_bias)), nn.Linear(dim, int(0.5*nodes)))
            # self.downsample = PatchMerging(input_resolution, dim, dim)
            # self.downsample = TopKPooling(dim, 0.25)
        else:
            self.downsample = None
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        ds = x
        # print('linear', self.layers[0][1].fn.net[0].weight.grad)
        
        if self.downsample:
            b, n, _ = x.size()
            observer = repeat(self.observer, '1 c d -> b c d', b = b)
            x = self.downsample(observer, context=x)
            # s = torch.softmax(s, dim=-1)
            # x = torch.matmul(s.transpose(1, 2), x) # B, k, D 
            # out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

            # link_loss = adj - torch.matmul(s, s.transpose(1, 2))
            # link_loss = torch.norm(link_loss, p=2)
            # if normalize is True:
            #     link_loss = link_loss / adj.numel()

            # ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()
            # print('downsample', self.downsample.select.weight.grad)
            # x, perm, score = self.downsample(x)
        return ds, x
'''  
class TFMEncoder(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, heads, dim_head, mlp_dim, dropout = 0., attn_dropout=0., downsample=False):
        super().__init__()
        self.layers = TFM(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout, attn_dropout=attn_dropout)
        if downsample:
            self.downsample = PatchMerging(input_resolution, dim, out_dim)
        else:
            self.downsample = None
    def forward(self, x):
        x = self.layers(x)
        ds = x
        # print('linear', self.layers[0][1].fn.net[0].weight.grad)
        
        if self.downsample:
            x = self.downsample(x)
        return ds, x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, heads, dim_head, mlp_dim, dropout = 0., attn_dropout=0., downsample=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        local_attn_heads = 0
        local_window_size = 256
        causal = False
        nb_features = None
        generalized_attention = False
        kernel_fn = nn.ReLU()
        no_projection = False
        qkv_bias = True
        attn_out_bias = True
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_attn_heads,
                                            local_window_size = local_window_size, nb_features = nb_features,
                                              generalized_attention = generalized_attention, kernel_fn = kernel_fn,
                                                dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias,
                                                  attn_out_bias = attn_out_bias)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        if downsample:
            self.downsample = PatchMerging(input_resolution, dim, out_dim)
        else:
            self.downsample = None
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        ds = x
        # print('linear', self.layers[0][1].fn.net[0].weight.grad)
        
        if self.downsample:
            x = self.downsample(x)
        return ds, x
    


class TransformerDecoder(nn.Module):
    def __init__(self, dim, in_dim, input_resolution, depth, heads, dim_head, mlp_dim, dropout = 0., attn_dropout=0., upsample=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        local_attn_heads = 0
        local_window_size = 256
        causal = False
        nb_features = None
        generalized_attention = True
        kernel_fn = nn.ReLU()
        # attn_dropout = 0.
        no_projection = False
        qkv_bias = True
        attn_out_bias = True
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_attn_heads,
                                             local_window_size = local_window_size, nb_features = nb_features,
                                               generalized_attention = generalized_attention, kernel_fn = kernel_fn,
                                               dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias,
                                                 attn_out_bias = attn_out_bias)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

        if upsample:
            self.upsample = PatchExpandLowerDim(input_resolution, in_dim)
        else:
            self.upsample = nn.Linear(in_dim, dim)
    def forward(self, x, context):
        # print(x.size())
        context = self.upsample(context)
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return x
    

class TransformerDec(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., attn_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        local_attn_heads = 0
        local_window_size = 256
        causal = False
        nb_features = None
        generalized_attention = True
        kernel_fn = nn.ReLU()
        # attn_dropout = 0.
        no_projection = False
        qkv_bias = True
        attn_out_bias = True
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_attn_heads,
                                             local_window_size = local_window_size, nb_features = nb_features,
                                               generalized_attention = generalized_attention, kernel_fn = kernel_fn,
                                               dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias,
                                                 attn_out_bias = attn_out_bias)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

        
    def forward(self, x, context):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return x


class ViP(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads,
                  mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., task='cls'):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert task in ['cls', 'sod'], 'Task must be either cls or sod'
        self.task = task
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.num_tokens = 256//dim

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim*self.num_tokens),
            nn.LayerNorm(dim*self.num_tokens),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # self.cls_token = nn.Parameter(torch.randn(1, self.num_tokens, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.locations = nn.Sequential(nn.Linear(22, dim*self.num_tokens), nn.LayerNorm(dim*self.num_tokens))

        
        self.transformer = Transformer(dim*self.num_tokens, depth, heads*self.num_tokens, dim_head, mlp_dim, emb_dropout, dropout, self.num_tokens)
        # self.transformer2 = Transformer(dim, block_depth, heads, dim_head, mlp_dim, emb_dropout, dropout)
        # self.transformer3 = Transformer(dim, block_depth, heads, dim_head, mlp_dim, emb_dropout, dropout)
        # self.transformer4 = Transformer(dim, block_depth, heads, dim_head, mlp_dim, emb_dropout, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.image_height = image_height
        self.image_width = image_width
        if self.task == 'sod':
            num_classes = 1
        else:
            num_classes = 1000 
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim*self.num_tokens),
            nn.Linear(dim*self.num_tokens, num_classes)
        )



    def forward(self, x):
        centroids = x[:, :, :2]
        fft = x[:, :, 8:-10]
        lbp = x[:, :,  -10:]
        color = x[:, :, 2:8]
        x = torch.cat((color, lbp), dim=2)
        locations = torch.cat((centroids, fft), dim=2)
        locations = self.locations(locations)


        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 c d -> b c d', b = b)

        x += locations
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # x = torch.cat(all_xs, dim=-1)
        # x = self.transformer2(x)
        # x = self.transformer3(x)
        # x = self.transformer4(x)

        if self.task == 'cls':
            x = x.mean(dim=1)# if self.pool == 'mean' else x[:, :4]
            # x = x.reshape(x.size(0), self.image_height//4, 4, self.image_width//4, 4, -1)
            # x = x.mean(dim=3).mean(dim=1)

            x = x.reshape(x.size(0), -1)
            x = self.to_latent(x)
            return self.mlp_head(x)
        else:
            x= self.to_latent(x)
            return self.mlp_head(x)
        
        


class ViPEncDec(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads,
                  mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
      
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 4, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.locations = nn.Sequential(nn.Linear(22, dim), nn.ReLU(), nn.Linear(dim, dim), nn.LayerNorm(dim))

        self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim, emb_dropout, dropout)
        self.transformer_dec = nn.ModuleList([TransformerDec(dim, 1, heads, dim_head, mlp_dim, emb_dropout, dropout) for _ in range(depth)])

        self.pool = pool
        self.to_latent = nn.Identity()

        
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )



    def forward(self, x):
        centroids = x[:, :, :2]
        fft = x[:, :, 8:-10]
        lbp = x[:, :,  -10:]
        color = x[:, :, 2:8]
        x = torch.cat((color, lbp), dim=2)

        locations = torch.cat((centroids, fft), dim=2)
        locations = self.locations(locations)


        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 c d -> b c d', b = b)

        x += locations
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer1(x)
        context = x
        for layer in self.transformer_dec:
            x = layer(x, context)

        
        x= self.to_latent(x)
        return self.head(x)
    




class ViPU(nn.Module):
    def __init__(self, *, image_size, patch_size, dims, depths, heads, mlp_ratio, 
                  channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels
     
        self.img_size = image_size
        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dims[0]),
            nn.LayerNorm(dims[0]),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.locations = nn.Sequential(nn.Linear(2, dims[0]), nn.LayerNorm(dims[0]))
        self.transformer_enc = nn.ModuleList([])
        resolutions = []
        for idx, depth in enumerate(depths):
            if idx == len(depths)-1:
                self.transformer_enc.append(TFMEncoder(dims[idx], dims[idx], (image_size//(2**idx), image_size//(2**idx)), depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout, downsample=False))
            elif idx < 2:
                self.transformer_enc.append(TransformerEncoder(dims[idx], dims[idx+1], (image_size//(2**idx), image_size//(2**idx)), depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout, downsample=True))
            else:
                self.transformer_enc.append(TFMEncoder(dims[idx], dims[idx+1], (image_size//(2**idx), image_size//(2**idx)), depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout, downsample=True))
            resolutions.append(image_size // (2 ** idx))
        # self.transformer_dec_1 = TransformerDecoder(dims[1], dims[2], (image_size//4, image_size//4), depths[1], heads[1], dims[1]//heads[1], int(mlp_ratio*dims[1]), emb_dropout, dropout, False)
        # self.transformer_dec_2 = TransformerDecoder(dims[0], dims[1], (image_size//2, image_size//2), depths[0], heads[0], dims[0]//heads[0], int(mlp_ratio*dims[0]), emb_dropout, dropout, False)
        # self.transformer_dec_3 = TransformerDecoder(dim, (image_size//2, image_size//2), depth, heads, dim_head, mlp_dim, emb_dropout, dropout, True)

        # dims.reverse()
        # self.upsample_layers = nn.ModuleList()
        # for i_layer in range(len(depths)):
        #     layer = BasicLayerUpsampleMA(dim=dims[i_layer],
        #                                total_dim=sum(dims),
        #                        input_resolution=resolutions,
        #                        num_heads=heads[len(depths)-i_layer-1],
        #                        mlp_ratio=4,
        #                        qkv_bias=True, 
        #                        qk_scale=1, 
        #                        drop=emb_dropout, 
        #                        attn_drop=dropout,
        #                        use_checkpoint=False)
        #     self.upsample_layers.append(layer)
        self.transformer_dec = nn.ModuleList([])
        depths_reverse = depths[::-1][1:]
        dims_reverse = dims[::-1]
        heads_reverse = heads[::-1]
       
        for idx, depth in enumerate(depths_reverse):
            if idx < len(depths_reverse)-2:
                self.transformer_dec.append(TFMDecoder(dims_reverse[idx+1], dims_reverse[idx],  depth,
                                    heads_reverse[idx+1], dims_reverse[idx+1]//heads_reverse[idx+1],
                                      int(mlp_ratio*dims_reverse[idx+1]),emb_dropout, dropout))
            else:
                self.transformer_dec.append(PerformerDecoder(dims_reverse[idx+1], dims_reverse[idx], depth,
                                    heads_reverse[idx+1], dims_reverse[idx+1]//heads_reverse[idx+1],
                                      int(mlp_ratio*dims_reverse[idx+1]),emb_dropout, dropout))
                

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dims[0]),
        #     nn.Linear(dims[0], 1)
        # )
        # self.upsample = nn.Upsample(size=image_size)
        # self.sod_head = nn.Linear(sum(dims), 1)
        self.sod_head = nn.Linear(dims_reverse[-1], 1)


    def forward(self, x):
        centroids = x[:, :, :2].float()
        # fft = x[:, :, 8:-10]
        # lbp = x[:, :,  -10:]
        # color = x[:, :, 2:8]
        # x = torch.cat((color, lbp), dim=2)
        
        x = x[:, :, 2:]

        # locations = torch.cat((centroids, fft), dim=2)
        locations = centroids
        
        locations = self.locations(locations)


        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        x += locations
  
        x = self.dropout(x)

        fts = []
        for idx, layer in enumerate(self.transformer_enc):

            ds, x = layer(x)
            fts.append(ds)
            
            
        fts.reverse()
        for idx, layer in enumerate(self.transformer_dec):
            x = layer(fts[idx+1], x)
        # up_ft = []
        # for idx, layer in enumerate(self.upsample_layers):
        #     x = layer(ft[len(ft)-idx-1], ft)
        #     ft[len(ft)-idx-1] = x
            
        #     res = int(math.sqrt(x.size(1)))
        #     x = x.reshape(x.size(0), res, res, -1).permute(0, 3, 1, 2)
        #     x = self.upsample(x).permute(0, 2, 3, 1)
        #     x = x.reshape(x.size(0), self.img_size**2, -1)
        #     up_ft.append(x)

        # up_ft = torch.cat(up_ft, dim=2)
        # print(x.size())

        # x = self.transformer_dec_1(feats[1], x)
        # x = self.transformer_dec_2(feats[0], x)
        # x = self.aspp(feats[-1])
        # x = self.upsample_layers(feats[0], feats[1], feats[2], x)
        
        x = self.sod_head(x)
        # x = self.transformer_dec(x, x)
        return x
        # return self.mlp_head(x)
    



class ViPUSLIC(nn.Module):
    def __init__(self, *, image_size, patch_size, dims, depths, heads, mlp_ratio, 
                  channels = 3, dropout = 0., emb_dropout = 0., num_seg, compactness):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        patch_dim = channels
     
        self.img_size = image_size
        self.compactness = compactness
        self.num_seg = num_seg
        self.to_slic = DiffSLIC(num_seg, n_iter=5, tau=0.01, candidate_radius=1, stable=True)
        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dims[0]),
            nn.LayerNorm(dims[0]),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.locations = nn.Sequential(nn.Linear(2, dims[0]), nn.LayerNorm(dims[0]))
        self.transformer_enc = nn.ModuleList([])
        resolutions = []
        for idx, depth in enumerate(depths):
            if idx == len(depths)-1:
                self.transformer_enc.append(TFMEncoder(dims[idx], dims[idx], (patch_size//(2**idx), patch_size//(2**idx)), depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout, downsample=False))
            elif idx < 2:
                self.transformer_enc.append(TransformerEncoder(dims[idx], dims[idx+1], (patch_size//(2**idx), patch_size//(2**idx)), depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout, downsample=True))
            else:
                self.transformer_enc.append(TFMEncoder(dims[idx], dims[idx+1], (patch_size//(2**idx), patch_size//(2**idx)), depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout, downsample=True))
            resolutions.append(patch_size // (2 ** idx))
       
        self.transformer_dec = nn.ModuleList([])
        depths.reverse()
        dims.reverse()
        heads.reverse()
        for idx, depth in enumerate(depths):
            if idx < len(depths)-2:
                self.transformer_dec.append(TFMDecoder(dims[idx], dims[max(idx-1, 0)],  depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout))
            else:
                self.transformer_dec.append(PerformerDecoder(dims[idx], dims[max(idx-1, 0)], depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout))
                
        
 
        self.sod_head = nn.Linear(dims[-1], 1)

    def batch_to_sparse(self, seg, img, mask):
        # seg (bs, 1, H, W)
        # img (bs, 3, H, W)
        # mask (bs, 1, H, W)

        b, _, h, w = img.size()
        xs = torch.arange(0, self.img_size, device=seg.device).unsqueeze(0).float()
        ys = torch.arange(0, self.img_size, device=seg.device).unsqueeze(1).float()
        xs = xs.repeat(self.img_size, 1)
        ys = ys.repeat(1, self.img_size)
        coord = torch.stack((xs, ys), 0).unsqueeze(0)
        coord = coord.repeat(b, 1, 1, 1)

        shift = torch.arange(0, b, device=seg.device).repeat_interleave(h*w)*self.num_seg
        
        seg = seg.reshape(-1)+shift
        img = img.permute(0, 2, 3, 1).reshape(-1, 3)
        mask = mask.reshape(-1)
        area = torch.ones_like(seg)
        
        coord = coord.permute(0, 2, 3, 1).reshape(-1, 2)
        # seg = SparseTensor(row=seg, col=torch.arange(0, seg.size(0), device='cuda'))
        # edge_index = torch.stack((torch.arange(0, seg.size(0), device='cuda'), seg), dim=0)
        
        
        colour = scatter(img, seg, reduce='mean', dim_size=self.num_seg*b)
        centroid = scatter(coord, seg, reduce='mean', dim_size=self.num_seg*b)
        area = scatter(area, seg, reduce='sum', dim_size=self.num_seg*b)
        seq_mask = scatter(mask, seg, reduce='mean', dim_size=self.num_seg*b)
        # colour = seg.matmul(img) # BS*H*W x 3
        # centroid = seg.matmul(coord) # BS*H*W x 2
        
        
        colour = colour.reshape(b, self.num_seg, 3)
        centroid = centroid.reshape(b, self.num_seg, 2)
        area = area.reshape(b, self.num_seg)
        seq_mask = seq_mask.reshape(b, self.num_seg)
        return colour, centroid, area, seq_mask


    def forward(self, img, mask):
        with torch.no_grad():
            x = img*2 - 1
            h, w = x.shape[-2:]
            coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, h, dtype=torch.float, device=x.device),
                                            torch.linspace(-1, 1, w, dtype=torch.float, device=x.device)), -1).unsqueeze(0)

            freqs = 2**torch.arange(2, dtype=torch.float, device=x.device)
            shape = coords.shape[:-1] + (-1,)
            scaled_x = (coords[..., None, :] * freqs[..., None]).reshape(shape) # (batch, *, n_points, num_feats * n_freq)
            scaled_x = torch.stack([scaled_x, scaled_x + 0.5 * torch.pi], -2).reshape(shape) # (batch, n_points, 2 * num_feats * n_freq)
            embedded_x = torch.sin(scaled_x).permute(0, 3, 1, 2) * self.compactness
            embedded_x = embedded_x.repeat(x.size(0), 1, 1, 1)
            # compute differentiable slic
            
            input = torch.cat([x, embedded_x], 1)
            feats, assign, p2s = self.to_slic(input)

            h_s, w_s = feats.shape[-2:]
        
            hard_assign = F.one_hot(assign.argmax(1), (2 * 1 + 1)**2).permute(0, 3, 1, 2).contiguous().float()
            
            label = torch.arange(h_s * w_s, dtype=torch.float, device=x.device).reshape(1, 1, h_s, w_s).repeat(x.size(0), 1, 1, 1)
            label = spixel_upsampling(label, hard_assign, candidate_radius=1).long()

            colour, centroids, area, seq_mask = self.batch_to_sparse(label, img, mask)
            # label_onehot = F.one_hot(label.reshape(x.size(0), -1), self.num_seg).float()
            # area = label_onehot.sum(1).unsqueeze(-1)
            # area_input = area.detach().clone()
    
            # As = label_onehot.permute(0, 2, 1)
            # Bs = ((x+1)/2.).reshape(x.size(0), 3, -1).permute(0, 2, 1)
            

            # xs = torch.arange(0, w, device=x.device).unsqueeze(0).float()
            # ys = torch.arange(0, h, device=x.device).unsqueeze(1).float()
            # xs = xs.repeat(h, 1)
            # ys = ys.repeat(1, w)
            # coord = torch.stack((xs, ys), 0).unsqueeze(0).repeat(x.size(0), 1, 1, 1)

            # Cs = coord.reshape(x.size(0), 2, -1).permute(0, 2, 1)
            # area[area==0] = torch.inf
            # colour = torch.clip(torch.einsum('bij,bjk->bik', As, Bs)/area, 0, 1)
            # centroids = torch.einsum('bij,bjk->bik', As, Cs)/area


        # x = torch.cat((colour, area_input), dim=2)
        # x = x.clone().detach()
        if self.training:
            x = colour.clone().detach().requires_grad_(True)
            centroids = centroids.clone().detach().requires_grad_(True)
        else:
            x = colour.clone().detach().requires_grad_(False)
            centroids = centroids.clone().detach().requires_grad_(False)


        locations = self.locations(centroids)

        x = self.to_patch_embedding(x)
 
        
        b, n, _ = x.shape

        x += locations
  
        x = self.dropout(x)

        fts = []
        for idx, layer in enumerate(self.transformer_enc):

            ds, x = layer(x)
            fts.append(ds)
            
            
        fts.reverse()
        for idx, layer in enumerate(self.transformer_dec):
            x = layer(fts[idx], x)
        # up_ft = []
        # for idx, layer in enumerate(self.upsample_layers):
        #     x = layer(ft[len(ft)-idx-1], ft)
        #     ft[len(ft)-idx-1] = x
            
        #     res = int(math.sqrt(x.size(1)))
        #     x = x.reshape(x.size(0), res, res, -1).permute(0, 3, 1, 2)
        #     x = self.upsample(x).permute(0, 2, 3, 1)
        #     x = x.reshape(x.size(0), self.img_size**2, -1)
        #     up_ft.append(x)

        # up_ft = torch.cat(up_ft, dim=2)
        # print(x.size())

        # x = self.transformer_dec_1(feats[1], x)
        # x = self.transformer_dec_2(feats[0], x)
        # x = self.aspp(feats[-1])
        # x = self.upsample_layers(feats[0], feats[1], feats[2], x)
        
        x = self.sod_head(x)
        # x = self.transformer_dec(x, x)
        return x, seq_mask, label
        # return self.mlp_head(x)
    


class SegViPUSLIC(nn.Module):
    def __init__(self, *, image_size, patch_size, dims, depths, heads, mlp_ratio, 
                  channels = 3, dropout = 0., emb_dropout = 0., num_seg, compactness):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        patch_dim = channels
     
        self.img_size = image_size
        self.compactness = compactness
        self.num_seg = num_seg
        # self.to_slic = DiffSLIC(num_seg, n_iter=5, tau=0.01, candidate_radius=1, stable=True)
        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dims[0]),
            nn.LayerNorm(dims[0])
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.locations = nn.Sequential(nn.Linear(2, dims[0]), nn.LayerNorm(dims[0]))
        self.transformer_enc = nn.ModuleList([])
        resolutions = []
        nodes = self.num_seg
        for idx, depth in enumerate(depths):
            nodes = int(nodes*0.25)
            if idx == len(depths)-1:
                self.transformer_enc.append(TFMEncoder(dims[idx], dims[idx], (patch_size//(2**idx), patch_size//(2**idx)), depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout, downsample=False))
            elif idx < 2:
                self.transformer_enc.append(TransformerEncoder(dims[idx], dims[idx+1], (patch_size//(2**idx), patch_size//(2**idx)), depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout, downsample=True))
            else:
                self.transformer_enc.append(TFMEncoder(dims[idx], dims[idx+1], (patch_size//(2**idx), patch_size//(2**idx)), depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout, downsample=True))
            resolutions.append(patch_size // (2 ** idx))
       
        self.transformer_dec = nn.ModuleList([])
        depths.reverse()
        dims.reverse()
        heads.reverse()
        for idx, depth in enumerate(depths):
            if idx < len(depths)-2:
                self.transformer_dec.append(TFMDecoder(dims[idx], dims[max(idx-1, 0)],  depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout))
            else:
                self.transformer_dec.append(PerformerDecoder(dims[idx], dims[max(idx-1, 0)], depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout))
                
        
        
        
        self.sod_head = nn.Linear(dims[-1], 1)


    def batch_to_sparse(self, seg, img, mask):
        # seg (bs, 1, H, W)
        # img (bs, 3, H, W)
        # mask (bs, 1, H, W)

        b, _, h, w = img.size()
        xs = torch.arange(0, self.img_size, device=seg.device).unsqueeze(0).float()
        ys = torch.arange(0, self.img_size, device=seg.device).unsqueeze(1).float()
        xs = xs.repeat(self.img_size, 1)
        ys = ys.repeat(1, self.img_size)
        coord = torch.stack((xs, ys), 0).unsqueeze(0)
        coord = coord.repeat(b, 1, 1, 1)

        shift = torch.arange(0, b, device=seg.device).repeat_interleave(h*w)*self.num_seg
        
        seg = seg.reshape(-1)+shift
        img = img.permute(0, 2, 3, 1).reshape(-1, 3)
        mask = mask.reshape(-1)
        area = torch.ones_like(seg)
        
        coord = coord.permute(0, 2, 3, 1).reshape(-1, 2)
        # seg = SparseTensor(row=seg, col=torch.arange(0, seg.size(0), device='cuda'))
        # edge_index = torch.stack((torch.arange(0, seg.size(0), device='cuda'), seg), dim=0)
        
        
        colour = scatter(img, seg, reduce='sum', dim_size=self.num_seg*b)
        centroid = scatter(coord, seg, reduce='mean', dim_size=self.num_seg*b)
        area = scatter(area, seg, reduce='sum', dim_size=self.num_seg*b)
        seq_mask = scatter(mask, seg, reduce='mean', dim_size=self.num_seg*b)
        # colour = seg.matmul(img) # BS*H*W x 3
        # centroid = seg.matmul(coord) # BS*H*W x 2
        
        
        colour = colour.reshape(b, self.num_seg, 3)
        centroid = centroid.reshape(b, self.num_seg, 2)
        area = area.reshape(b, self.num_seg)
        seq_mask = seq_mask.reshape(b, self.num_seg)
        return colour, centroid, area, seq_mask


    def forward(self, seg, img, mask):
        with torch.no_grad():
            
            seg = seg.long()
            
            colour, centroids, area, seq_mask = self.batch_to_sparse(seg, img, mask)

        # import  matplotlib.pyplot as plt
        # from skimage.segmentation import mark_boundaries
        # fig, ax = plt.subplots(1, 2)
        # # plt.imshow(mark_boundaries(img[0].permute(1, 2, 0).detach().numpy(), result.labels.reshape(1, 224, 224).detach().numpy().squeeze()))
        # ax[0].imshow(colour[0].reshape(1024, 3).detach().cpu().numpy()[seg[0].detach().cpu().numpy().reshape(-1), :].reshape(224, 224, 3))
        # ax[0].scatter(centroids[0][:, 0].detach().cpu().numpy(), centroids[0][:, 1].detach().cpu().numpy(), c='red', s=10)
        # ax[1].imshow(seq_mask[0].reshape(1024, 1).detach().cpu().numpy()[seg[0].detach().cpu().numpy().reshape(-1), :].reshape(224, 224, 1), cmap='gray')
        # plt.show()

        
        x = colour.clone().detach().requires_grad_(True)
        
        centroids = centroids.clone().detach().requires_grad_(True)
        locations = self.locations(centroids)
        
        x = self.to_patch_embedding(x)
        
        
        b, n, _ = x.shape

        x += locations
  
        x = self.dropout(x)

        fts = []
        for idx, layer in enumerate(self.transformer_enc):

            ds, x = layer(x)
            fts.append(ds)
            
       
        fts.reverse()
        for idx, layer in enumerate(self.transformer_dec):
            x = layer(fts[idx], x)
       
        x = self.sod_head(x)
        # x = self.transformer_dec(x, x)
        
        return x, seq_mask
        # return self.mlp_head(x)
    
    

class ViPEnc(nn.Module):
    def __init__(self, *, image_size, patch_size, dims, depths, heads, mlp_ratio, 
                  channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels
     

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dims[0]),
            nn.LayerNorm(dims[0]),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.locations = nn.Sequential(nn.Linear(22, dims[0]), nn.LayerNorm(dims[0]))

        
        self.transformer_enc = nn.ModuleList([])
        nodes = image_size*image_size
        for idx, depth in enumerate(depths):
            if idx == len(depths)-1:
                self.transformer_enc.append(TFMEncoder(dims[idx], dims[idx], (image_size//(2**idx), image_size//(2**idx)), depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout, downsample=False))
            elif idx < 2:
                self.transformer_enc.append(TransformerEncoder(dims[idx], dims[idx+1], (image_size//(2**idx), image_size//(2**idx)), depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout, downsample=True))
            else:
                self.transformer_enc.append(TFMEncoder(dims[idx], dims[idx+1], (image_size//(2**idx), image_size//(2**idx)), depth,
                                    heads[idx], dims[idx]//heads[idx], int(mlp_ratio*dims[idx]),emb_dropout, dropout, downsample=True))


            

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], 1000)
        )



    def forward(self, x):
        centroids = x[:, :, :2]
        fft = x[:, :, 8:-10]
        lbp = x[:, :,  -10:]
        color = x[:, :, 2:8]
        x = torch.cat((color, lbp), dim=2)
        
        locations = torch.cat((centroids, fft), dim=2)
        locations = self.locations(locations)


        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        x += locations
  
        x = self.dropout(x)

        for layer in self.transformer_enc:
            ds, x = layer(x)
            # print(x.size())
        # x1, x = self.transformer_enc_1(x)
        # x2, x = self.transformer_enc_2(x)
        # x = self.transformer_enc_3(x)

        
        
        x = x.mean(dim = 1)
        return self.mlp_head(x)
