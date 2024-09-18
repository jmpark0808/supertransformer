import math

import torch
from torch import nn, einsum
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack

from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb

# constant

TOKEN_SELF_ATTN_VALUE = -5e4

# helper functions

def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(tensor):
    dtype = tensor.dtype
    normed = F.normalize(tensor, dim = -1)
    return normed.type(dtype)

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if seqlen % multiple == 0:
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim = dim)

# main class

class WindowSampling(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, in_dim, heads, head_dim, K,  qk_scale=None, attn_drop=0.):

        super().__init__()
        
        self.fmt = nn.Parameter(torch.randn(1, heads, K, head_dim))
        self.heads = heads

        self.k = nn.Linear(in_dim, heads*head_dim)
        self.v = nn.Linear(in_dim, heads*head_dim)
        self.proj = nn.Linear(heads*head_dim, heads*head_dim)
        self.K = K


        self.scale = qk_scale or 1.0
        
        self.attn_drop = nn.Dropout(attn_drop)
   
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
         
        q = self.fmt.repeat(x.size(0), 1, 1, 1)
        k = self.k(x).reshape(x.size(0), x.size(1), self.heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).reshape(x.size(0), x.size(1), self.heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) # B, H, K, N

       
        attn = self.softmax(attn) 

        attn = self.attn_drop(attn)

        x = (attn @ v) # B, H, K, D
       
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), self.K, -1)
        x = self.proj(x)
        return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, window_size, qk_scale=None, attn_drop=0.):

        super().__init__()
        self.window_size = window_size  # Wh, Ww

        self.scale = qk_scale or 1.0
        
        self.attn_drop = nn.Dropout(attn_drop)
   
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
  
        B, H, N, C = q.shape
        height = int(N**0.5)
        width = int(N**0.5)
        q = q.reshape(B, H, height, width, C)
        q = q.view(B, H, height // self.window_size, self.window_size, width // self.window_size, self.window_size, C)
        q = q.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, H, self.window_size*self.window_size, C)

        k = k.reshape(B, H, height, width, C)
        k = k.view(B, H, height // self.window_size, self.window_size, width // self.window_size, self.window_size, C)
        k = k.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, H, self.window_size*self.window_size, C)

        v = v.reshape(B, H, height, width, C)
        v = v.view(B, H, height // self.window_size, self.window_size, width // self.window_size, self.window_size, C)
        v = v.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, H, self.window_size*self.window_size, C) # BWW, H, L*L, C


        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

       
        attn = self.softmax(attn) # BWW, H, L*L, L*L

        attn = self.attn_drop(attn)

        x = (attn @ v) # BWW, H, L*L, C 
       
        x = x.view(B, height // self.window_size, width // self.window_size, H, self.window_size, self.window_size, -1) # B, W, W, H, L, L, C
        x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, H, N, -1) # B, N (=L*L*W*W), C*H  -> B, H, N (=L*L*W*W), C
        return x



class DilatedAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, window_size, qk_scale=None, attn_drop=0.):

        super().__init__()
        self.window_size = window_size  # Wh, Ww

        self.scale = qk_scale or 1.0
        
        self.attn_drop = nn.Dropout(attn_drop)
   
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
  
        B, H, N, C = q.shape
        height = int(N**0.5)
        width = int(N**0.5)
        q = q.reshape(B, H, height, width, C)
        q = q.view(B, H, self.window_size, height // self.window_size, self.window_size, width // self.window_size,  C)
        q = q.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, H, self.window_size*self.window_size, C)

        k = k.reshape(B, H, height, width, C)
        k = k.view(B, H, self.window_size, height // self.window_size,  self.window_size, width // self.window_size,  C)
        k = k.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, H, self.window_size*self.window_size, C)

        v = v.reshape(B, H, height, width, C)
        v = v.view(B, H, self.window_size, height // self.window_size, self.window_size, width // self.window_size,  C)
        v = v.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, H, self.window_size*self.window_size, C) # BWW, H, L*L, C


        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

       
        attn = self.softmax(attn) # BWW, H, L*L, L*L

        attn = self.attn_drop(attn)

        x = (attn @ v) # BWW, H, L*L, C 
       
        x = x.view(B, height // self.window_size, width // self.window_size, H, self.window_size, self.window_size, -1) # B, W, W, H, L, L, C
        x = x.permute(0, 3, 4, 1, 5, 2, 6).contiguous().view(B, H, N, -1) # B, N (=L*L*W*W), C*H  -> B, H, N (=L*L*W*W), C
        return x




class GlobalAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, qk_scale=None, attn_drop=0.):

        super().__init__()

        self.scale = qk_scale or 1.0
        
        self.attn_drop = nn.Dropout(attn_drop)
   
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
  
        B, H, N, C = q.shape
        

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

       
        attn = self.softmax(attn) # B, H, N, N

        attn = self.attn_drop(attn)

        x = (attn @ v) # B, H, N, C 
       
        # x = x.permute(0, 2, 1, 3, 1, 4, 2, 5, 6).contiguous().view(B, H, N, -1) # B, N (=L*L*W*W), C*H  -> B, H, N (=L*L*W*W), C
        return x





class LocalAttention(Module):
    def __init__(
        self,
        window_size,
        causal = False,
        look_backward = 1,
        look_forward = None,
        dropout = 0.,
        shared_qk = False,
        rel_pos_emb_config = None,
        dim = None,
        autopad = False,
        exact_windowsize = False,
        scale = None,
        use_rotary_pos_emb = True,
        use_xpos = False,
        xpos_scale_base = None
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'

        self.scale = scale

        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize

        self.causal = causal

        self.look_backward = look_backward
        self.look_forward = look_forward

        self.dropout = nn.Dropout(dropout)

        self.shared_qk = shared_qk

        # relative positions

        self.rel_pos = None
        self.use_xpos = use_xpos

        if use_rotary_pos_emb and (exists(rel_pos_emb_config) or exists(dim)):  # backwards compatible with old `rel_pos_emb_config` deprecated argument
            if exists(rel_pos_emb_config):
                dim = rel_pos_emb_config[0]

            self.rel_pos = SinusoidalEmbeddings(
                dim,
                use_xpos = use_xpos,
                scale_base = default(xpos_scale_base, window_size // 2)
            )

    def forward(
        self,
        q, k, v,
        mask = None,
        input_mask = None,
        attn_bias = None,
        window_size = None
    ):

        mask = default(mask, input_mask)

        assert not (exists(window_size) and not self.use_xpos), 'cannot perform window size extrapolation if xpos is not turned on'

        shape, autopad, pad_value, window_size, causal, look_backward, look_forward, shared_qk = q.shape, self.autopad, -1, default(window_size, self.window_size), self.causal, self.look_backward, self.look_forward, self.shared_qk

        # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))

        # auto padding

        if autopad:
            orig_seq_len = q.shape[1]
            (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (q, k, v))

        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype

        scale = default(self.scale, dim_head ** -0.5)

        assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'

        windows = n // window_size

        if shared_qk:
            k = l2norm(k)

        seq = torch.arange(n, device = device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)

        # bucketing

        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))

        bq = bq * scale

        look_around_kwargs = dict(
            backward =  look_backward,
            forward =  look_forward,
            pad_value = pad_value
        )

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        # rotary embeddings

        if exists(self.rel_pos):
            pos_emb, xpos_scale = self.rel_pos(bk)
            bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale = xpos_scale)

        # calculate positions for masking

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')

        pad_mask = bq_k == pad_value

        sim = einsum('b h i e, b h j e -> b h i j', bq, bk)

        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert (b % heads) == 0

            attn_bias = repeat(attn_bias, 'h i j -> (b h) 1 i j', b = b // heads)
            sim = sim + attn_bias

        mask_value = max_neg_value(sim)

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, TOKEN_SELF_ATTN_VALUE)
            del self_mask

        if causal:
            causal_mask = bq_t < bq_k

            if self.exact_windowsize:
                max_causal_window_size = (self.window_size * self.look_backward)
                causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))

            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        # masking out for exact window size for non-causal
        # as well as masking out for padding value

        if not causal and self.exact_windowsize:
            max_backward_window_size = (self.window_size * self.look_backward)
            max_forward_window_size = (self.window_size * self.look_forward)
            window_mask = ((bq_k - max_forward_window_size) > bq_t) | (bq_t > (bq_k + max_backward_window_size)) | pad_mask
            sim = sim.masked_fill(window_mask, mask_value)
        else:
            sim = sim.masked_fill(pad_mask, mask_value)

        # take care of key padding mask passed in

        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0

            h = b // mask.shape[0]

            if autopad:
                _, mask = pad_to_multiple(mask, window_size, dim = -1, value = False)

            mask = rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size)
            mask = look_around(mask, **{**look_around_kwargs, 'pad_value': False})
            mask = rearrange(mask, '... j -> ... 1 j')
            mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregation

        out = einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(out, 'b w n d -> b (w n) d')

        if autopad:
            out = out[:, :orig_seq_len, :]

        out, *_ = unpack(out, packed_shape, '* n d')
        return out