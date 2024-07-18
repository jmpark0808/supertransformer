import torch
import torch.nn as nn
from performer_pytorch import SelfAttention
from einops import rearrange, repeat
import math

class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1):
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
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
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        # y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection
        y = self.dp(self.proj(y))
        return y

    def forward(self, x):
        x = x + self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x




class Performer(nn.Module):
    def __init__(self, input_dim, embed_dim, heads, depth, num_classes):
        super().__init__()
        # self.performer = perf(
        #         dim = embed_dim,
        #         depth = depth,
        #         heads = heads,
        #         dim_head = embed_dim,
        #         causal = False
        # )
        # self.performer = nn.Sequential(*[Token_performer(embed_dim, embed_dim, head_cnt=heads) for _ in range(depth)])
        self.performer = nn.Sequential(*[SelfAttention(dim=embed_dim, heads=heads, dim_head=embed_dim//heads) for _ in range(depth)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.locations = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.LayerNorm(embed_dim),
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