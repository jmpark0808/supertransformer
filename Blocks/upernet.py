import torch
import torch.nn as nn
from Blocks.merge import PatchExpand
import math
from Blocks.swin_rope import SwinTransformerBlockRoPE

class SwinDecoder(nn.Module):
    def __init__(self, low_level_idx, high_level_idx, 
                 input_size, input_dim,input_high_dim, input_middle_dim, num_classes,
                 depth, last_layer_depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate, norm_layer, decoder_norm, use_checkpoint):
        super().__init__()
        self.low_level_idx = low_level_idx # 0
        self.high_level_idx = high_level_idx # 2

        self.proj_high = nn.Linear(input_high_dim, input_dim,bias=False) # 通道从384转换成96
        self.proj_middle = nn.Linear(input_middle_dim,input_dim,bias=False) # 通道数从192转换成96

        self.layers_up = nn.ModuleList()
        for i in range(high_level_idx - low_level_idx):# 0 , 1
            layer_up = BasicLayer_up(dim=int(input_dim),# 输入的通道是96
                                    input_resolution=(input_size*2**i, input_size*2**i),
                                    depth=depth,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=drop_path_rate,
                                    norm_layer=norm_layer,
                                    upsample=PatchExpand,
                                    use_checkpoint=use_checkpoint)
            
            self.layers_up.append(layer_up)
        self.upsample = BasicLayer_up(dim=int(input_dim),# 输入的通道是96
                                    input_resolution=(input_size*2, input_size*2),
                                    depth=depth,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=drop_path_rate,
                                    norm_layer=norm_layer,
                                    upsample=PatchExpand,
                                    use_checkpoint=use_checkpoint)

        # self.last_layers_up = nn.ModuleList()
        # for _ in range(low_level_idx+1): # 1
        #     i+=1
        #     last_layer_up = BasicLayer_up(dim=int(input_dim)*3, # 96 * 3
        #                                     input_resolution=(input_size*2**i, input_size*2**i),
        #                                     depth=last_layer_depth,
        #                                     num_heads=num_heads,
        #                                     window_size=window_size,
        #                                     mlp_ratio=mlp_ratio,
        #                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                     drop=drop_rate, attn_drop=attn_drop_rate,
        #                                     drop_path=0.0,
        #                                     norm_layer=norm_layer,
        #                                     upsample=PatchExpand,
        #                                     use_checkpoint=use_checkpoint)
        #     self.last_layers_up.append(last_layer_up)
        
        i += 1
        # self.final_up = PatchExpand(input_resolution=(input_size*2**i, input_size*2**i),
        #                             dim=int(input_dim)*3,
        #                             dim_scale=2,
        #                             norm_layer=norm_layer)
        
        if decoder_norm: # True
            self.norm_up = norm_layer(int(input_dim)*3)
        else:
            self.norm_up = None
        self.output = nn.Conv2d(int(input_dim)*3, num_classes, kernel_size=1, bias=False)

    def forward(self, low_level,middle_level,high_level, aspp):
        """
        low_level: B, Hl, Wl, C
        aspp: B, Ha, Wa, C
        """
        high_trans = self.proj_high(high_level) # 14x14x96
        target = high_trans + aspp # 14x14x96
        middle = self.proj_middle(middle_level) # 28x28x96
        B,HM,WM,MC = middle.shape
        middle = middle.view(B,HM*WM,MC)

        B, Hl, Wl, C = low_level.shape
        _, Ha, Wa, _ = aspp.shape
        # _,hm,wm,mc = middle_level.shape
        # _,hh,hw,hc = high_level.shape
        _,ht,wt,ct = target.shape
        low_level = low_level.view(B, Hl*Wl, C) # 56 * 56 * 96
        # middle = middle_level.view(B,hm*wm,mc) # 28 * 28 * 192
        # high = high_level.view(B,hh*hw,hc)   # 14*14*384
        aspp = aspp.view(B, Ha*Wa, C)    # 14 * 14 * 96
        target = target.view(B,ht*wt,ct)
        index = 0
        up = None
        for layer in self.layers_up:
            target = layer(target) # ASPP的特征先经过上采样 最后得到 56×56×96 的特征图
            if index == 0:
                up = target # 28x28x96
                target = target + middle # 28x28x96
                index = 1
        up_1 = self.upsample(up) # 56x56x96
        up_2 = target # 56x56x96
        up_3 = target + low_level 

        x = torch.cat([up_1,up_2,up_3], dim=-1) # 在通道维数上进行拼接 56×56×192

        # for layer in self.last_layers_up:
        #     x = layer(x) # 上采样到 112 × 112 × 192

        if self.norm_up is not None: #True
            x = self.norm_up(x)
            
        # x = self.final_up(x) # 放大到 225 × 225 × 192  
    
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.output(x) # 得到了概率分布图  225 × 225 × 9
        
        return x
    
class SwinDecoderNoASPP(nn.Module):
    def __init__(self, low_level_idx, high_level_idx, 
                 input_size, input_dim,input_high_dim, input_middle_dim, num_classes,
                 depth, last_layer_depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate, norm_layer, decoder_norm, use_checkpoint):
        super().__init__()
        self.low_level_idx = low_level_idx # 0
        self.high_level_idx = high_level_idx # 2

        self.proj_high = nn.Linear(input_high_dim, input_dim,bias=False) # 通道从384转换成96
        self.proj_middle = nn.Linear(input_middle_dim,input_dim,bias=False) # 通道数从192转换成96

        self.layers_up = nn.ModuleList()
        for i in range(high_level_idx - low_level_idx):# 0 , 1
            layer_up = BasicLayer_up(dim=int(input_dim),# 输入的通道是96
                                    input_resolution=(input_size*2**i, input_size*2**i),
                                    depth=depth,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=drop_path_rate,
                                    norm_layer=norm_layer,
                                    upsample=PatchExpand,
                                    use_checkpoint=use_checkpoint)
            
            self.layers_up.append(layer_up)
        self.upsample = BasicLayer_up(dim=int(input_dim),# 输入的通道是96
                                    input_resolution=(input_size*2, input_size*2),
                                    depth=depth,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=drop_path_rate,
                                    norm_layer=norm_layer,
                                    upsample=PatchExpand,
                                    use_checkpoint=use_checkpoint)

        # self.last_layers_up = nn.ModuleList()
        # for _ in range(low_level_idx+1): # 1
        #     i+=1
        #     last_layer_up = BasicLayer_up(dim=int(input_dim)*3, # 96 * 3
        #                                     input_resolution=(input_size*2**i, input_size*2**i),
        #                                     depth=last_layer_depth,
        #                                     num_heads=num_heads,
        #                                     window_size=window_size,
        #                                     mlp_ratio=mlp_ratio,
        #                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                     drop=drop_rate, attn_drop=attn_drop_rate,
        #                                     drop_path=0.0,
        #                                     norm_layer=norm_layer,
        #                                     upsample=PatchExpand,
        #                                     use_checkpoint=use_checkpoint)
        #     self.last_layers_up.append(last_layer_up)
        
        i += 1
        # self.final_up = PatchExpand(input_resolution=(input_size*2**i, input_size*2**i),
        #                             dim=int(input_dim)*3,
        #                             dim_scale=2,
        #                             norm_layer=norm_layer)
        
        if decoder_norm: # True
            self.norm_up = norm_layer(int(input_dim)*3)
        else:
            self.norm_up = None
        self.output = nn.Conv2d(int(input_dim)*3, num_classes, kernel_size=1, bias=False)

    def forward(self, low_level,middle_level,high_level):
        """
        low_level: B, Hl, Wl, C
        aspp: B, Ha, Wa, C
        """
        high_trans = self.proj_high(high_level) # 14x14x96
        target = high_trans
        middle = self.proj_middle(middle_level) # 28x28x96
        B,HM,WM,MC = middle.shape
        middle = middle.view(B,HM*WM,MC)

        B, Hl, Wl, C = low_level.shape
        # _,hm,wm,mc = middle_level.shape
        # _,hh,hw,hc = high_level.shape
        _,ht,wt,ct = target.shape
        low_level = low_level.view(B, Hl*Wl, C) # 56 * 56 * 96
        # middle = middle_level.view(B,hm*wm,mc) # 28 * 28 * 192
        # high = high_level.view(B,hh*hw,hc)   # 14*14*384
      
        target = target.view(B,ht*wt,ct)
        index = 0
        up = None
        for layer in self.layers_up:
            target = layer(target) # ASPP的特征先经过上采样 最后得到 56×56×96 的特征图
            if index == 0:
                up = target # 28x28x96
                target = target + middle # 28x28x96
                index = 1
        up_1 = self.upsample(up) # 56x56x96
        up_2 = target # 56x56x96
        up_3 = target + low_level 

        x = torch.cat([up_1,up_2,up_3], dim=-1) # 在通道维数上进行拼接 56×56×192

        # for layer in self.last_layers_up:
        #     x = layer(x) # 上采样到 112 × 112 × 192

        if self.norm_up is not None: #True
            x = self.norm_up(x)
            
        # x = self.final_up(x) # 放大到 225 × 225 × 192  
    
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.output(x) # 得到了概率分布图  225 × 225 × 9
        
        return x

class BasicLayer_up(nn.Module):
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

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

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
                                 norm_layer=norm_layer)
            for i in range(depth)]) # 尺度是不变的

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x
    


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x) :
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x) :
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result],1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, input_dim, reduction, input_size, out_dim):
        super().__init__()
        self.input_size = input_size
        self.ca = ChannelAttention(channel=input_dim, reduction=reduction) # 通道注意力
        self.sa = SpatialAttention(kernel_size=1)  # 空间注意力
 
        self.proj = nn.Linear(input_dim, out_dim)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        B, L, C = x.shape
        assert L == self.input_size ** 2
        x = x.permute(0, 2, 1).contiguous()# B C L
        x = x.view(B, C, self.input_size, self.input_size)
        
        residual = x
        out = x * self.ca(x) # 通道注意力机制 通道变成32了
        out = out * self.sa(out) # 空间注意力
        out = out + residual
        out = out.view(B, C, L).permute(0, 2, 1).contiguous()
        return self.proj(out)



class SwinASPP(nn.Module):
    def __init__(self, input_size, input_dim, out_dim, cross_attn,
                 depth, num_heads, mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate, 
                 norm_layer, aspp_norm, aspp_activation, start_window_size,
                 aspp_dropout, downsample, use_checkpoint):
        
        super().__init__()
        
        self.out_dim = out_dim # 96
        if input_size == 24:
            self.possible_window_sizes = [4, 6, 8, 12, 24]
        else:
            # l-level 就是14  2-level就是7,14  3-level就是2,7,14  4-level就是1,2,7,14
            self.possible_window_sizes = [i for i in range(start_window_size, input_size+1) if input_size%i==0] 

        self.layers = nn.ModuleList()
        for ws in self.possible_window_sizes:
            layer = BasicLayerDownsample(dim=int(input_dim), # 384
                               input_resolution=(input_size, input_size),
                               depth=1 if ws==input_size else depth,
                               num_heads=num_heads, # 3
                               window_size=ws,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=drop_path_rate,
                               norm_layer=norm_layer,
                               downsample=downsample,
                               use_checkpoint=use_checkpoint)  # # 除了最后一个窗口是1，其他都是2
            
            self.layers.append(layer)
        
        if cross_attn == 'CBAM':
            self.proj = CBAMBlock(input_dim=len(self.layers)*input_dim, 
                                  reduction=12, 
                                  input_size=input_size,
                                  out_dim=out_dim)
        else:
            self.proj = nn.Linear(len(self.layers)*input_dim, out_dim)
        
        # Check if needed
        self.norm = norm_layer(out_dim) if aspp_norm else None
        if aspp_activation == 'relu':
            self.activation = nn.ReLU()
        elif aspp_activation == 'gelu':
            self.activation = nn.GELU()
        elif aspp_activation is None:
            self.activation = None
        
        self.dropout = nn.Dropout(aspp_dropout)

    def forward(self, x):
        """
        x: input tensor (high level features) with shape (batch_size, input_size, input_size, input_dim)

        returns ...
        """
        B, H, W, C = x.shape
        x = x.view(B, H*W, C)

        features = []
        for layer in self.layers:
            out, _ = layer(x)
            features.append(out)

        features = torch.cat(features, dim=-1)
        features = self.proj(features) # 输出的维度变成了96

        # Check if needed 
        if self.norm is not None:
            features = self.norm(features)
        if self.activation is not None:
            features = self.activation(features)
        features = self.dropout(features)

        return features.view(B, H, W, self.out_dim)
    