import torch
import torch.nn as nn
from performer_pytorch import SelfAttention, CrossAttention
from einops import rearrange, repeat
import math

class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1):
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, in_dim)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(dim, 1 * dim),
            nn.GELU(),
            nn.Linear(1 * dim, dim),
            nn.Dropout(dp2),
        )

        self.m = int(in_dim * kernel_ratio)
        self.w = torch.randn(self.m, in_dim)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, 1, self.m) / 2

        wtx = torch.einsum('bthi,mi->bthm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        k, q, v = k.reshape(k.size(0), k.size(1), self.head_cnt, -1), q.reshape(q.size(0), q.size(1), self.head_cnt, -1), v.reshape(v.size(0), v.size(1), self.head_cnt, -1), 
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, h, m), (B, T, h, m)
        D = torch.einsum('bthi,bhi->bth', qp, kp.sum(dim=1)).unsqueeze(dim=3)  # (B, T, h, m) * (B, h, m) -> (B, T, h, 1)
        kptv = torch.einsum('bihn,bihm->bhnm', v.float(), kp)  # (B, h, emb/h, m)
        y = torch.einsum('bthi,bhni->bthn', qp, kptv) / (D.repeat(1, 1, 1, self.emb//self.head_cnt) + self.epsilon)  # (B, T, h, emb)/Diag
        # skip connection
        # y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection
        y = y.reshape(y.size(0), y.size(1), -1)
        y = self.dp(self.proj(y))
        return y

    def forward(self, x):
        x = x + self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PerformerBlock(nn.Module):
    def __init__(self, dim, heads, attn_dropout, dropout, mlp_ratio):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
            nn.Dropout(dropout),
        )

        self.layer = SelfAttention(dim=dim, heads=heads, dim_head=dim//heads, dropout=attn_dropout)


    def forward(self, x):
        x = x + self.layer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class PerformerDecoderBlock(nn.Module):
    def __init__(self, dim, heads, attn_dropout, dropout, mlp_ratio):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
            nn.Dropout(dropout),
        )

        self.layer = CrossAttention(dim=dim, heads=heads, dim_head=dim//heads, dropout=attn_dropout)


    def forward(self, q, kv):
        kv = kv + self.layer(self.norm1(q), context=self.norm1(kv))
        kv = kv + self.mlp(self.norm2(kv))
        return kv



class Performer(nn.Module):
    def __init__(self, input_dim, embed_dim, heads, depth, num_classes, attn_dropout, dropout, mlp_ratio):
        super().__init__()
        # self.performer = perf(
        #         dim = embed_dim,
        #         depth = depth,
        #         heads = heads,
        #         dim_head = embed_dim,
        #         causal = False
        # )
        self.performer = nn.Sequential(*[Token_performer(embed_dim, embed_dim, head_cnt=heads) for _ in range(depth)])
        # self.performer = nn.Sequential(*[PerformerBlock(dim=embed_dim, heads=heads,
        #                                                  attn_dropout=attn_dropout,
        #                                                    dropout=dropout, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim), 
        )
        self.locations = nn.Sequential(
            nn.Linear(2, embed_dim),
        )

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        centroids = x[:, :, :2]
        fft = x[:, :, 8:-10]
        lbp = x[:, :,  -10:]
        color = x[:, :, 2:8]
        x = torch.cat((color, lbp, fft), dim=2)
        
        locations = self.locations(centroids)

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x += locations
        x = torch.cat((cls_tokens, x), dim=1)
        
        # x = self.dropout(x)
        
        x = self.performer(x)

        x = x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    


class PerformerU(nn.Module):
    def __init__(self, input_dim, embed_dim, heads, depth, attn_dropout, dropout, mlp_ratio):
        super().__init__()
        # self.performer = perf(
        #         dim = embed_dim,
        #         depth = depth,
        #         heads = heads,
        #         dim_head = embed_dim,
        #         causal = False
        # )
        self.performer = nn.Sequential(*[Token_performer(embed_dim, embed_dim, head_cnt=heads) for _ in range(depth)])
        # self.performer_enc = nn.Sequential(*[PerformerBlock(dim=embed_dim, heads=heads,
        #                                                  attn_dropout=attn_dropout,
        #                                                    dropout=dropout, mlp_ratio=mlp_ratio) for _ in range(depth)])
        # self.performer_dec = nn.ModuleList([PerformerDecoderBlock(dim=embed_dim, heads=heads,
        #                                                  attn_dropout=attn_dropout,
        #                                                    dropout=dropout, mlp_ratio=mlp_ratio) for _ in range(depth)])
        
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim), 
        )
        self.locations = nn.Sequential(
            nn.Linear(2, embed_dim),
        )

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(embed_dim, 1)

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
        
        x = self.performer(x)
        # kv = None
        # for dec in self.performer_dec:
        #     if kv is None:
        #         kv = dec(x, x)
        #     else:
        #         kv = dec(x, kv)

        kv = x
        kv = self.to_latent(kv)
        return self.mlp_head(kv)