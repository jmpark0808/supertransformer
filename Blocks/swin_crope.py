import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from Blocks.swin_common import window_partition, window_reverse, Mlp
from typing import Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


WindowProcess = None
WindowProcessReverse = None
print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

def init_random_2d_freqs(head_dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    theta = theta
    mag = 1 / (theta ** (torch.arange(0, head_dim, 4)[: (head_dim // 4)].float() / head_dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs


def compute_cis(freqs, t_x, t_y):
    N = t_x.shape[0]
    # a = torch.randn(64, 49, 1)
    # b = torch.randn(64, 3, 1, 8)
    # c = torch.einsum('bac,bdce->bdae', a, b)
    # print(c.size())
    # No float 16 for this range
    # freqs = freqs.repeat(t_x.size(0), 1, 1)
    b = t_x.size(0)
    with torch.amp.autocast('cuda', enabled=False):
        freqs_x = torch.einsum('bac,bdce->bdae',t_x.unsqueeze(-1), freqs[0].unsqueeze(-2).unsqueeze(0).repeat(b, 1, 1, 1)) #(t_x @ freqs[0].unsqueeze(-2))
        freqs_y = torch.einsum('bac,bdce->bdae',t_y.unsqueeze(-1), freqs[1].unsqueeze(-2).unsqueeze(0).repeat(b, 1, 1, 1)) #(t_y @ freqs[1].unsqueeze(-2))
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
        
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim

    # assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    window_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # print(xq_.size(), freqs_cis.size())
    # freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    if window_mask is not None:
        window_mask = window_mask.unsqueeze(-1).unsqueeze(1).repeat(1, xq_.size(1), 1, xq_.size(3))
        xq_out = torch.where(window_mask == 0, xq_ * freqs_cis, xq_)
        xk_out = torch.where(window_mask == 0, xk_ * freqs_cis, xk_)
    else:
        xq_out = xq_ * freqs_cis
        xk_out = xk_ * freqs_cis
    xq_out = torch.view_as_real(xq_out).flatten(3)
    xk_out = torch.view_as_real(xk_out).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

class WindowAttentionRoPE(nn.Module):
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rope_div_factor=1, rope_theta = 10.0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.rdf = rope_div_factor



        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # if dim < 48:


        # freqs = init_random_2d_freqs(
        #     head_dim=self.dim // self.num_heads, num_heads=self.num_heads, theta=rope_theta, 
        #     rotate=True
        # )
        self.pos = nn.Linear(2, head_dim)

    def forward(self, x, centroids, mask=None, window_mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        
        # if self.dim < 48:
        if window_mask is not None:
            window_mask_centroids = window_mask.repeat(q.size(0)//window_mask.size(0), 1)
            
            # Push all the centroids that we don't care about to 1e9 (part of the shifted window)
            centroids_x = torch.where(window_mask_centroids == 0, centroids[:, :, 1], 1000)
            centroids_y = torch.where(window_mask_centroids == 0, centroids[:, :, 0], 1000)

            min_centroids_x = torch.min(centroids_x, dim=1, keepdim=True).values
            min_centroids_y = torch.min(centroids_y, dim=1, keepdim=True).values
            
            t_x = (centroids_x - min_centroids_x)
            t_y = (centroids_y - min_centroids_y)

            window_mask_centroids = torch.where(centroids[:, :, 0] == 1000, 10, window_mask_centroids)
        else:
            window_mask_centroids = torch.where(centroids[:, :, 0] == 1000, 10, 0)
            min_centroids_x = torch.min(centroids[:, :, 1], dim=1, keepdim=True).values
            min_centroids_y = torch.min(centroids[:, :, 0], dim=1, keepdim=True).values
            t_x = (centroids[:, :, 1] - min_centroids_x)
            t_y = (centroids[:, :, 0] - min_centroids_y)

        
        # if not self.training:
            
        #     altered_t_x = torch.where(t_x<1e6, t_x, -1e9)
        #     altered_t_y = torch.where(t_y<1e6, t_y, -1e9)
        #         # print('shifted', torch.min(t_x), torch.max(altered_t_x), torch.min(t_y), torch.max(altered_t_y))


        #     if len(torch.argwhere(torch.max(altered_t_x, dim=1).values>10).squeeze()) >0:
        #         indices = torch.argwhere(torch.max(altered_t_x, dim=1).values>10).squeeze()
        #         for ind in indices:
        #             plt.scatter(altered_t_x[ind].detach().cpu().numpy(), altered_t_y[ind].detach().cpu().numpy())
        #             plt.show()
        #     t_x_mins = torch.min(t_x, dim=1).values
        #     t_x_maxs = torch.max(altered_t_x, dim=1).values
        #     t_y_mins = torch.min(t_y, dim=1).values
        #     t_y_maxs = torch.max(altered_t_y, dim=1).values

        #     plt.scatter(t_x_mins.detach().cpu().numpy(), t_y_mins.detach().cpu().numpy())
        #     plt.scatter(t_x_maxs.detach().cpu().numpy(), t_y_maxs.detach().cpu().numpy())
        #     plt.show()
        #     if torch.max(t_x) > 10:
        #         for i in range(t_x.size(0)):

        #             plt.scatter(t_x[i].detach().cpu().numpy(), t_y[i].detach().cpu().numpy())
        #             # for x, y, t in zip(t_x[i].detach().cpu().numpy(), t_y[i].detach().cpu().numpy(), [str(o) for o in range(len(np.squeeze(t_x[i].detach().cpu().numpy())))]):
        #             #     plt.text(x, y, t)
        #         plt.title(f'{self.dim}')
        #         plt.show()
        # if not self.training:
           

        #     plt.scatter(t_x[t_x<500].detach().cpu().numpy(),
        #                 t_y[t_y<500].detach().cpu().numpy())
        #     plt.show()

        # freqs_cis = compute_cis(self.rope_freqs, t_x, t_y)

        # q, k = apply_rotary_emb(q, k, freqs_cis, window_mask_centroids)
        centroids_feat = torch.stack((t_x, t_y), dim=2) # B, N, 2
       
        centroids_feat = self.pos(centroids_feat).unsqueeze(1) #B, 1, N, D
        
        q = q + centroids_feat
        k = k + centroids_feat
        attn = (q @ k.transpose(-2, -1))


        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # x = torch.mean(x, 2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

   

class SwinTransformerBlockRoPE(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False, rope_div_factor=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionRoPE(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, rope_div_factor=rope_div_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, [self.window_size, self.window_size])  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1e9)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
            mask_windows = None

        self.register_buffer("mask", mask_windows)
        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x, centroids):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        centroids = centroids.view(B, 2, H, W).permute(0, 2, 3, 1)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, [self.window_size, self.window_size])  # nW*B, window_size, window_size, C

                shifted_centroids = torch.roll(centroids, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                centroid_windows = window_partition(shifted_centroids, [self.window_size, self.window_size]) 
                # partition windows
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            shifted_centroids = centroids
            # partition windows
            x_windows = window_partition(shifted_x, [self.window_size, self.window_size])  # nW*B, window_size, window_size, C
            centroid_windows = window_partition(shifted_centroids, [self.window_size, self.window_size])

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        centroid_windows = centroid_windows.view(-1, self.window_size*self.window_size, 2)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, centroid_windows, mask=self.attn_mask, window_mask=self.mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, [self.window_size, self.window_size], H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, [self.window_size, self.window_size], H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

   




class PatchMergingRoPE(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, centroids):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        c0 = centroids[:, :, 0::2, 0::2]  # B 2, H/2 W/2 
        c1 = centroids[:, :, 1::2, 0::2]  # B 2, H/2 W/2 
        c2 = centroids[:, :, 0::2, 1::2]  # B 2, H/2 W/2 
        c3 = centroids[:, :, 1::2, 1::2]  # B 2, H/2 W/2 
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        centroids_stack = torch.stack([c0, c1, c2, c3], 1)

        mask = centroids_stack < 1000
        # import matplotlib.pyplot as plt

        # fig, ax = plt.subplots(1, 2)
        # ax[0].scatter(centroids[0, 1, :, :].detach().cpu().numpy(), centroids[0, 0, :, :].detach().cpu().numpy())
        
        centroids = torch.where(mask.sum(dim=1) == 0, 1000, (mask*centroids_stack).sum(dim=1) / mask.sum(dim=1))
        if torch.sum(torch.isnan(centroids))> 0:
            assert 0, 'Merging centroids cause NaNs'
        
        x = self.norm(x)
        x = self.reduction(x)

        return x, centroids

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    


class BasicLayerRoPE(nn.Module):
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
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False, rope_div_factor=1):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlockRoPE(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process,
                                 rope_div_factor = rope_div_factor)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, centroids):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, centroids)
        ds = x
        if self.downsample is not None:
            x, centroids = self.downsample(x, centroids)
        return ds, x, centroids

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"





