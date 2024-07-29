import torch, math
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from math import ceil
from functools import partial
from contextlib import contextmanager
from Blocks.swin_common import PatchMerging, PatchExpand

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
        attn_out_bias = True
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)

        self.heads = heads
        self.global_heads = heads - local_heads
        # self.local_attn = LocalAttention(window_size = local_window_size, causal = causal, autopad = True, dropout = dropout, look_forward = int(not causal), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None

        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias = attn_out_bias)
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
        out =  self.to_out(out)
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
                PreNorm(dim, SelfAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_attn_heads,
                                            local_window_size = local_window_size, nb_features = nb_features,
                                              generalized_attention = generalized_attention, kernel_fn = kernel_fn,
                                                dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias,
                                                  attn_out_bias = attn_out_bias)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    


class TransformerEncoder(nn.Module):
    def __init__(self, dim, input_resolution, depth, heads, dim_head, mlp_dim, dropout = 0., attn_dropout=0., downsample=False):
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
            # self.downsample = PatchMerging(input_resolution, dim, dim)
            self.downsample = SelectTopK(dim, input_resolution)
        else:
            self.downsample = None
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        ds = x
        if self.downsample:
            x, perm = self.downsample(x)
        return ds, x
    

class TransformerDecoder(nn.Module):
    def __init__(self, dim, input_resolution, depth, heads, dim_head, mlp_dim, dropout = 0., attn_dropout=0., upsample=False):
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
            self.upsample = PatchExpand(input_resolution, dim, dim_scale=2)
        else:
            self.upsample = None
    def forward(self, x, context):
        
        if self.upsample:
            x = self.upsample(x)
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

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 4, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.locations = nn.Sequential(nn.Linear(2, dim), nn.LayerNorm(dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, emb_dropout, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        if self.task == 'sod':
            num_classes = 1
        else:
            num_classes = 1000 
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )



    def forward(self, x):
        centroids = x[:, :, :2]
        fft = x[:, :, 8:-10]
        lbp = x[:, :,  -10:]
        color = x[:, :, 2:8]
        x = torch.cat((color, lbp, fft), dim=2)
        
        locations = self.locations(centroids)


        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 c d -> b c d', b = b)

        x += locations
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.task == 'cls':
            x = x.mean(dim = 1) # if self.pool == 'mean' else x[:, :4]

            # x = x.reshape(x.size(0), -1)
            x = self.to_latent(x)
            return self.mlp_head(x)
        else:
            x= self.to_latent(x)
            return self.mlp_head(x)
    



class ViPU(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, 
                  channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels
     

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.locations = nn.Sequential(nn.Linear(2, dim), nn.LayerNorm(dim))

        self.transformer_enc_1 = TransformerEncoder(dim, (image_size, image_size), depth, heads, dim_head, mlp_dim,
                                                     emb_dropout, dropout, downsample=True)
        self.transformer_enc_2 = TransformerEncoder(dim, (image_size//2, image_size//2), depth, heads, dim_head, mlp_dim,
                                                     emb_dropout, dropout, downsample=True)
        self.transformer_enc_3 = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout,
                                                                                               batch_first=True, norm_first=True),
                                                             num_layers=depth*3)
        # self.transformer_enc_3 = TransformerEncoder(dim, (image_size//4, image_size//4), depth*3, heads, dim_head, mlp_dim,
        #                                              emb_dropout, dropout, downsample=False)
        
        self.transformer_dec_1 = TransformerDecoder(dim, (image_size//4, image_size//4), depth, heads, dim_head, mlp_dim, emb_dropout, dropout, False)
        self.transformer_dec_2 = TransformerDecoder(dim, (image_size//4, image_size//4), depth, heads, dim_head, mlp_dim, emb_dropout, dropout, False)
        # self.transformer_dec_3 = TransformerDecoder(dim, (image_size//2, image_size//2), depth, heads, dim_head, mlp_dim, emb_dropout, dropout, True)


        

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )



    def forward(self, x):
        centroids = x[:, :, :2]
        fft = x[:, :, 8:-10]
        lbp = x[:, :,  -10:]
        color = x[:, :, 2:8]
        x = torch.cat((color, lbp, fft), dim=2)
        
        locations = self.locations(centroids)


        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        x += locations
  
        x = self.dropout(x)

        x1, x = self.transformer_enc_1(x)
        x2, x = self.transformer_enc_2(x)
        x = self.transformer_enc_3(x)


        x = self.transformer_dec_1(x2, x)
        x = self.transformer_dec_2(x1, x)
        

        # x = self.transformer_dec(x, x)

        return self.mlp_head(x)
    


class SelectTopK(nn.Module):
    r"""Selects the top-:math:`k` nodes with highest projection scores from the
    `"Graph U-Nets" <https://arxiv.org/abs/1905.05178>`_, `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
    and `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ papers.

    If :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`, computes:

        .. math::
            \mathbf{y} &= \sigma \left( \frac{\mathbf{X}\mathbf{p}}{\|
            \mathbf{p} \|} \right)

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

    If :obj:`min_score` :math:`\tilde{\alpha}` is a value in :obj:`[0, 1]`,
    computes:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\mathbf{X}\mathbf{p})

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

    where :math:`\mathbf{p}` is the learnable projection vector.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): The graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        act (str or callable, optional): The non-linearity :math:`\sigma`.
            (default: :obj:`"tanh"`)
    """
    def __init__(
        self,
        in_channels,
        input_resolution,
        ):
        super().__init__()

        
        self.in_channels = in_channels
        self.input_resolution = input_resolution

        self.weight = torch.nn.Parameter(torch.randn(1, 1, in_channels))

        

    

    def forward(
        self,
        x

        ):
        """"""  # noqa: D419
        
        H, W = self.input_resolution
        score = (x * self.weight).sum(dim=-1)

        
        score = torch.tanh(score / self.weight.norm(p=2, dim=-1))
        
        score = score.view(score.size(0), -1)
        topk, perm = torch.topk(score, H*W//4)
        
        topk = torch.gather(x, 1, perm.unsqueeze(-1).repeat(1, 1, x.size(-1)))
        

        return topk, perm
    

class ViPEnc(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, 
                  channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels
     

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.locations = nn.Sequential(nn.Linear(2, dim), nn.LayerNorm(dim))

        self.transformer_enc_1 = TransformerEncoder(dim, (image_size, image_size), depth, heads, dim_head, mlp_dim,
                                                     emb_dropout, dropout, downsample=True)
        self.transformer_enc_2 = TransformerEncoder(dim, (image_size//2, image_size//2), depth, heads, dim_head, mlp_dim,
                                                     emb_dropout, dropout, downsample=True)
        self.transformer_enc_3 = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout,
                                                                                               batch_first=True, norm_first=True),
                                                             num_layers=depth*3)
        

        

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1000)
        )



    def forward(self, x):
        centroids = x[:, :, :2]
        fft = x[:, :, 8:-10]
        lbp = x[:, :,  -10:]
        color = x[:, :, 2:8]
        x = torch.cat((color, lbp, fft), dim=2)
        
        locations = self.locations(centroids)


        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        x += locations
  
        x = self.dropout(x)

        x1, x = self.transformer_enc_1(x)
        x2, x = self.transformer_enc_2(x)
        x = self.transformer_enc_3(x)


        
        x = x.mean(dim = 1)
        return self.mlp_head(x)
