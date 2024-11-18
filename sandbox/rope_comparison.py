
import torch
from typing import Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def init_t_xy(end_x: int, end_y: int, zero_center=False):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    
    return t_x, t_y

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
    # No float 16 for this range
    with torch.amp.autocast('cuda', enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
        
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

def compute_cis_crope(freqs, t_x, t_y):
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

t_x, t_y = init_t_xy(end_x=9, end_y=9)
plt.scatter(t_x, t_y)
plt.show()
freqs = init_random_2d_freqs(
            head_dim=16, num_heads=2, theta=10, 
            rotate=True
        )
freqs_cis = compute_cis(freqs, t_x, t_y)
features = np.load('/home/eddie/Datasets/sp_train/n02113624_4906.npy')

centroids = torch.tensor(features[:, :2])

centroids = centroids.reshape(1, 72, 72, 2)
B, H, W, C = centroids.shape
centroids = centroids.view(B, H // 9, 9, W // 9, 9, C)
centroids = centroids.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, 9* 9 , C)

min_centroids_x = torch.min(centroids[:, :, 1], dim=1, keepdim=True).values
min_centroids_y = torch.min(centroids[:, :, 0], dim=1, keepdim=True).values
t_x = (centroids[:, :, 1] - min_centroids_x)/4
t_y = (centroids[:, :, 0] - min_centroids_y)/4
plt.scatter(t_x, t_y)
plt.show()

freqs_cis_crope = compute_cis_crope(freqs, t_x, t_y)

for head in range(2):
    for hd in range(8):
        data = freqs_cis[head,:,hd]
        x = data.real
        y = data.imag
        plt.plot(x, y)
        for i in range(64):
            data = freqs_cis_crope[i, head,:,hd]
            x = data.real
            y = data.imag
            plt.plot(x, y)

        plt.show()
        

# q, k = apply_rotary_emb(q, k, freqs_cis)



