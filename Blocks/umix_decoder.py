from Blocks.swin_rope import init_t_xy, init_random_2d_freqs, compute_cis, apply_rotary_emb_sep
import torch.nn as nn
import torch
from Blocks.merge import Mlp

class UMixDecoder(nn.Module):
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

    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., q_res=None, kv_res=None):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_ln = nn.LayerNorm(dim)
        self.kv_ln = nn.LayerNorm(dim)
        self.ln = nn.LayerNorm(dim)


        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.mlp = Mlp(dim, int(dim*mlp_ratio), drop=proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        q_t_x, q_t_y = init_t_xy(end_x=q_res, end_y=q_res)
        self.register_buffer('rope_q_t_x', q_t_x)
        self.register_buffer('rope_q_t_y', q_t_y)

        kv_t_x, kv_t_y = init_t_xy(end_x=kv_res, end_y=kv_res)
        self.register_buffer('rope_kv_t_x', kv_t_x)
        self.register_buffer('rope_kv_t_y', kv_t_y)

        freqs = init_random_2d_freqs(
            head_dim=self.dim // self.num_heads, num_heads=self.num_heads, theta=10, 
            rotate=True
        )
        
        self.rope_freqs = nn.Parameter(freqs, requires_grad=True)


    def forward(self, q, kv):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N_q, C = q.shape

        q = self.q_ln(q)
        q_skip = q

        kv = self.kv_ln(kv)
        
        freqs_cis_q = compute_cis(self.rope_freqs, self.rope_q_t_x, self.rope_q_t_y)
        freqs_cis_kv = compute_cis(self.rope_freqs, self.rope_kv_t_x, self.rope_kv_t_y)


        q = self.q(q).reshape(B_, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        B_, N_kv, C = kv.shape
        
        kv = self.kv(kv).reshape(B_, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]

        q = q * self.scale
        q, k = apply_rotary_emb_sep(q, k, freqs_cis_q, freqs_cis_kv)
        
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N_q, C)

        x += q_skip
        x = self.ln(x)
        x_skip = x
        
        x = self.mlp(x)

        x += x_skip
        return x

class UMixDecoderCROPE(nn.Module):
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

    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_ln = nn.LayerNorm(dim)
        self.kv_ln = nn.LayerNorm(dim)
        self.ln = nn.LayerNorm(dim)


        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.mlp = Mlp(dim, int(dim*mlp_ratio), drop=proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, kv):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N_q, C = q.shape

        q = self.q_ln(q)
        q_skip = q

        kv = self.kv_ln(kv)
        

        q = self.q(q).reshape(B_, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        B_, N_kv, C = kv.shape
        
        kv = self.kv(kv).reshape(B_, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N_q, C)

        x += q_skip
        x = self.ln(x)
        x_skip = x
        
        x = self.mlp(x)

        x += x_skip
        return x


class BasicLayerUpsampleMA(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, total_dim, q_resolution, input_resolution, num_heads, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=1.0, attn_drop=0., drop=0.,  use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint
        # build blocks
        
        # self.blocks = nn.TransformerDecoderLayer(dim,nhead=num_heads, dim_feedforward=int(mlp_ratio*dim), 
        #                                          dropout=drop, batch_first=True, norm_first=True) 
        min_res_0 = min([res[0] for res in input_resolution])
        min_res_1 = min([res[1] for res in input_resolution])
        self.blocks = UMixDecoder(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, attn_drop, drop , q_resolution, min_res_0)
        
        self.avg_pools = nn.ModuleList([nn.AvgPool2d((res[0]//min_res_0, res[1]//min_res_1), (res[0]//min_res_0, res[1]//min_res_1)) for res in input_resolution])
        
        self.linear = nn.Linear(total_dim, dim)
        

       
    def forward(self, x_q, x_kv):
        feats = []
        for idx,(h, w) in enumerate(self.input_resolution):
            feats.append(self.avg_pools[idx](x_kv[idx].reshape(x_kv[idx].size(0), h, w, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))

        # feat1 = self.avg_pool_x8(x_kv[0].reshape(x_kv[0].size(0), 32, 32, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # feat2 = self.avg_pool_x4(x_kv[1].reshape(x_kv[1].size(0), 16, 16, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # feat3 = self.avg_pool_x2(x_kv[2].reshape(x_kv[2].size(0), 8, 8, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # feat4 = x_kv[3].reshape(x_kv[3].size(0), 4, 4, -1)

        features = torch.cat(feats, dim=3)
        features = features.reshape(features.size(0), -1, features.size(3))
        features = self.linear(features)
        
        out = self.blocks(x_q, features)
        

        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
    

class BasicLayerUpsampleMACROPE(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, total_dim, input_resolution, num_heads, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=1.0, attn_drop=0., drop=0.,  use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = UMixDecoder(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, attn_drop, drop )
        # self.blocks = nn.TransformerDecoderLayer(dim,nhead=num_heads, dim_feedforward=int(mlp_ratio*dim), 
        #                                          dropout=drop, batch_first=True, norm_first=True) 
        min_res_0 = min([res[0] for res in input_resolution])
        min_res_1 = min([res[1] for res in input_resolution])
        
        self.avg_pools = nn.ModuleList([nn.AvgPool2d((res[0]//min_res_0, res[1]//min_res_1), (res[0]//min_res_0, res[1]//min_res_1)) for res in input_resolution])
        
        self.linear = nn.Linear(total_dim, dim)
        

       
    def forward(self, x_q, x_kv, x_q_centroids, x_kv_centroids):
        feats = []
        for idx,(h, w) in enumerate(self.input_resolution):
            feats.append(self.avg_pools[idx](x_kv[idx].reshape(x_kv[idx].size(0), h, w, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))

        # feat1 = self.avg_pool_x8(x_kv[0].reshape(x_kv[0].size(0), 32, 32, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # feat2 = self.avg_pool_x4(x_kv[1].reshape(x_kv[1].size(0), 16, 16, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # feat3 = self.avg_pool_x2(x_kv[2].reshape(x_kv[2].size(0), 8, 8, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # feat4 = x_kv[3].reshape(x_kv[3].size(0), 4, 4, -1)

        features = torch.cat(feats, dim=3)
        features = features.reshape(features.size(0), -1, features.size(3))
        features = self.linear(features)
        
        out = self.blocks(x_q, features, x_q_centroids, x_kv_centroids)
        

        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops