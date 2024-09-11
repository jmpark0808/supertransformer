import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Callable, Optional, Tuple

def update_clst_feats(elem_feats: torch.Tensor,
                      clst_feats: torch.Tensor,
                      stride: Tuple[int, int],
                      tau: float=0.01,
                      candidate_radius: int=1,
                      stable: bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""update cluster features with a local attention. this function equivalent to
    compute_center_to_elem_assignment with spixel_downsampling

    Args:
        elem_feats (torch.Tensor): a tensor of shape (batch, channels, height, width)
                                   height and width should be larger than those of clst_feats
        clst_feats (torch.Tensor): a tensor of shape (batch, channels, height_c, width_c)
        stride (Tuple[int, int]): grid size when dividing elem_feats into height_c * width_c grids
        tau (float): a temperature parameter.
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled
        stable (bool): if True, using stable compuatation of softmax withe temperature

    Returns:
        new_clst_feats (torch.Tensor): a tensor of shape (batch, channels, height_c, width_c)
        soft_assignment (torch.Tensor): a tensor of shape
                                        (batch, stride_h * stride_w * (2*candidate_radius + 1)**2, height_c, width_c)
                                        each element has a non-negative value
        similarities (torch.Tensor): a tensor of shape
                                     (batch, stride_h * stride_w * (2*candidate_radius + 1)**2, height_c, width_c)
                                     a similarity matrix having real values
    """
    b, c, h, w = clst_feats.shape # initial centers
    neighbor_range = candidate_radius * 2 + 1 # 3
    kernel_size = (stride[0]*neighbor_range, stride[1]*neighbor_range) #(30, 30)
    padding = (stride[0]*candidate_radius, stride[1]*candidate_radius) #(10, 10)
    n_candidate_pixels = kernel_size[0] * kernel_size[1] # 900
    
    unfold_elem_feats = F.unfold(elem_feats, kernel_size, padding=padding, stride=stride) # 2700 x 1024
    unfold_elem_feats = unfold_elem_feats.reshape(b, c, n_candidate_pixels, h, w) # 1 x 5 x 900 x 32 x 32
    
    similarities = torch.einsum('bcphw,bchw->bphw', (unfold_elem_feats, clst_feats)) # 1 x 900 x 32 x 32
    similarities = torch.where(similarities==0, -torch.inf, similarities)
    if stable:
        similarities = similarities - similarities.max(1, keepdim=True).values.detach()
    soft_assignment = torch.softmax(similarities / tau, dim=1)
    new_clst_feats = torch.einsum('bphw,bcphw->bchw', (soft_assignment, unfold_elem_feats))
    return new_clst_feats, soft_assignment

def compute_elem_to_center_assignment(clst_feats: torch.Tensor,
                                      elem_feats: torch.Tensor,
                                      stride: Tuple[int, int],
                                      tau: float=0.01,
                                      candidate_radius: int=1,
                                      stable: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""compute elem-to-center assignment with a local attention

    Args:
        clst_feats (torch.Tensor): a tensor of shape (batch, channels, height_c, width_c)
        elem_feats (torch.Tensor): a tensor of shape (batch, channels, height, width)
                                   height and width should be larger than those of clst_feats
        stride (Tuple[int, int]): grid size when dividing elem_feats into height_c * width_c grids
        tau (float): a temperature parameter.
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled
        stable (bool): if True, using stable compuatation of softmax withe temperature

    Returns:
        soft_assignment (torch.Tensor): a tensor of shape
                                        (batch, (2*candidate_radius + 1)**2, height, width)
                                        each element has a non-negative value
        similarities (torch.Tensor): a tensor of shape
                                     (batch, (2*candidate_radius + 1)**2, height, width)
                                     a similarity matrix having real values
    """
    batch_size, channels, height, width = elem_feats.shape # original data
    n_spixels = clst_feats.shape[2] * clst_feats.shape[3] # 1024
    neighbor_range = candidate_radius * 2 + 1 # 3
    candidate_clusters = F.unfold(clst_feats, kernel_size=neighbor_range, padding=candidate_radius)
    candidate_clusters = candidate_clusters.reshape(batch_size, channels, neighbor_range**2, n_spixels) # 1 x 3 x 9 x 1024
    unfold_elem_feats = F.unfold(elem_feats, kernel_size=stride, stride=stride)
    unfold_elem_feats = unfold_elem_feats.reshape(batch_size, channels, stride[0] * stride[1], n_spixels) # 1 x 3 x 100 x 1024
    similarities = torch.einsum('bkcn,bkpn->bcpn', (candidate_clusters, unfold_elem_feats))
    similarities = similarities.contiguous().reshape(batch_size * neighbor_range**2, -1, n_spixels) # 9 x 100 x 1024
    similarities = F.fold(similarities, (height, width), kernel_size=stride, stride=stride)
    similarities = similarities.reshape(batch_size, neighbor_range**2, height, width) # 1 x 9 x 320 x 320
    # masking zero padding regions with -inf
    # by using the fact that the inner product is zero.
    similarities = torch.where(similarities==0, -torch.inf, similarities)
    if stable:
        similarities = similarities - similarities.max(1, keepdim=True).values.detach()
    soft_assignment = (similarities / tau).softmax(1) # 1 x 9 x 320 x 320
    superpixel_indices = torch.arange(1, n_spixels+1).reshape(1, 1, clst_feats.shape[2], clst_feats.shape[3]).repeat(batch_size, 1, 1, 1).float()
    superpixel_indices = F.unfold(superpixel_indices, kernel_size=neighbor_range, padding=candidate_radius) 
    superpixel_indices = superpixel_indices.reshape(batch_size, neighbor_range**2, clst_feats.shape[2], clst_feats.shape[3]) # 1 x 9 x 32 x 32
    superpixel_indices = torch.repeat_interleave(torch.repeat_interleave(superpixel_indices, stride[0], dim=2), stride[1], dim=3) # 1 x 9 x 320 x 320
    soft_assignment_argmax = torch.argmax(soft_assignment, dim=1, keepdim=True) # 1 x 1, 320 x 320
    get_hard_assignment = torch.gather(superpixel_indices, 1, soft_assignment_argmax)
   
    return soft_assignment, similarities, get_hard_assignment

class DiffSLIC(nn.Module):
    r"""Differentiable SLIC

    Args:
        n_spixels (int): a number of superpixels
        n_iter (int): a number of iterations for updating cluster centers
        tau (float): a temperature parameter. when tau -> 0, assignemnt is deterministic
        normalize (bool): if True, pixel and superpixel features are normalized so that those l2 norm are 1
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled
        stable (bool): if True, using stable compuatation of softmax with temperature
                       `stable` should be True, when using extremely small tau for obtaining deterministic assignment.
    """
    def __init__(self,
                 n_spixels: int,
                 n_iter: int=5,
                 tau: float=0.01,
                 candidate_radius: int=1,
                 normalize: bool=True,
                 stable: bool=False) -> None:
        super().__init__()
        self.n_spixels = n_spixels
        self.n_iter = n_iter
        self.tau = tau
        self.candidate_radius = candidate_radius
        self.normalize = normalize
        self.stable = stable
        

    def forward(self, x: torch.Tensor,
                clst_feats: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Args:
            x (torch.Tensor): a tensor of shape (batch, channels, height, width)
            clst_feats (Optional[torch.Tensor]): a tensor of shape (batch, channels, height_s, width_s)
                                                 initial cluster features. if clst_feats is None, it is
                                                 initialized by averaging pixels in a uniform grid

        Returns:
            clst_feats (torch.Tensor): a tensor of shape (batch, channels, height_s, width_s)
                                       height_s * width_s <= self.n_spixels
            p2s_assign (torch.Tensor): a tensor of shape (batch, (2*candidate_radius + 1)**2, height, width)
                                       a pixel-to-superpixel assignemnt matrix
            s2p_assign (torch.Tensor): a tensor of shape
                                       (batch, stride_h * stride_w * (2*candidate_radius + 1)**2, height_s, width_s)
                                       a superpixel-to-pixel assignemnt matrix
                                       if n_iter is 0, s2p_assign is None
        """
        height, width = x.shape[-2:]
        # xs = torch.arange(0, width).unsqueeze(0).float()/width
        # ys = torch.arange(0, height).unsqueeze(1).float()/height
        # xs = xs.repeat(height, 1)
        # ys = ys.repeat(1, width)
        # coord = torch.stack((xs, ys), 0).unsqueeze(0).repeat(x.size(0), 1, 1, 1)
        
        # x = torch.cat((x, coord), dim=1)
        # initialize cluster features
        if clst_feats is None:
            height_s = int(math.sqrt(self.n_spixels * height / width))
            width_s = int(math.sqrt(self.n_spixels * width / height))
            stride_h = (height + height_s - 1) // height_s
            stride_w = (width + width_s - 1) // width_s
            stride = (stride_h, stride_w)
            clst_feats = F.adaptive_avg_pool2d(x, (height_s, width_s))
        else:
            height_s, width_s = clst_feats.shape[-2:]
            stride = ((height + height_s) // height_s, (width + width_s) // width_s)
        # normalize feature vectors so that their l2-norm is 1
        if self.normalize:
            x = x / x.norm(dim=1, keepdim=True)
            clst_feats = clst_feats / clst_feats.norm(dim=1, keepdim=True)
        # padding an image feature so that its height and width are divisible by stride values
        pad_x = (width_s - width % width_s) % width_s
        pad_y = (height_s - height % height_s) % height_s

        
        x = F.pad(x, (0, pad_x, 0, pad_y))


        # update cluster features
        s2p_assign = None
        for _ in range(self.n_iter):
            clst_feats, s2p_assign = update_clst_feats(x, clst_feats, stride, self.tau, self.candidate_radius)
            if self.normalize:
                clst_feats = clst_feats / clst_feats.norm(dim=1, keepdim=True)
        # compute a pixel-to-superpixel assignment
        p2s_assign, _ , get_hard_assignment= compute_elem_to_center_assignment(clst_feats, x, stride, self.tau, self.candidate_radius)
        # remove the padding region
        if pad_y > 0:
            p2s_assign = p2s_assign[..., :-pad_y, :]
        if pad_x > 0:
            p2s_assign = p2s_assign[..., :-pad_x]
        return clst_feats, p2s_assign, s2p_assign, get_hard_assignment

    def extra_repr(self):
        return f'n_spixels={self.n_spixels}, \n ' \
               f'n_iter={self.n_iter}, \n ' \
               f'tau={self.tau}, \n ' \
               f'candidate_radius={self.candidate_radius}, \n' 
