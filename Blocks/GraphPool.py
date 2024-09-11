import torch


from typing import Callable, Optional, Tuple

from torch import Tensor
import random
from torch_geometric.data import Batch, Data

from torch_geometric.nn.pool.pool import pool_batch, pool_edge, pool_pos
from torch_geometric.utils import add_self_loops, scatter

def consecutive_cluster(src):
    unique, inv = torch.unique(src, sorted=True, return_inverse=True)
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    return inv, perm



def _max_pool_x(
    cluster: Tensor,
    x: Tensor,
    size: Optional[int] = None,
) -> Tensor:
    
    return scatter(x, cluster, dim=0, dim_size=size, reduce='max')

def select(x: Tensor,
           perm: Tensor) -> Tensor:
    
    return x[perm]

def max_pool_x(
    cluster: Tensor,
    x: Tensor,
    batch: Tensor,
    batch_size: Optional[int] = None,
    size: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Max-Pools node features according to the clustering defined in
    :attr:`cluster`.

    Args:
        cluster (torch.Tensor): The cluster vector
            :math:`\mathbf{c} \in \{ 0, \ldots, N - 1 \}^N`, which assigns each
            node to a specific cluster.
        x (Tensor): The node feature matrix.
        batch (torch.Tensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
        size (int, optional): The maximum number of clusters in a single
            example. This property is useful to obtain a batch-wise dense
            representation, *e.g.* for applying FC layers, but should only be
            used if the size of the maximum number of clusters per example is
            known in advance. (default: :obj:`None`)

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`) if :attr:`size` is
        :obj:`None`, else :class:`torch.Tensor`
    """
    if size is not None:
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1
        return _max_pool_x(cluster, x, batch_size * size), None

    cluster, perm = consecutive_cluster(cluster)
    x = _max_pool_x(cluster, x)
    batch = pool_batch(perm, batch)

    return x, batch


def max_pool(
    cluster: Tensor,
    data: Data,
    transform: Optional[Callable] = None,
) -> Data:
    r"""Pools and coarsens a graph given by the
    :class:`torch_geometric.data.Data` object according to the clustering
    defined in :attr:`cluster`.
    All nodes within the same cluster will be represented as one node.
    Final node features are defined by the *maximum* features of all nodes
    within the same cluster, node positions are averaged and edge indices are
    defined to be the union of the edge indices of all nodes within the same
    cluster.

    Args:
        cluster (torch.Tensor): The cluster vector
            :math:`\mathbf{c} \in \{ 0, \ldots, N - 1 \}^N`, which assigns each
            node to a specific cluster.
        data (Data): Graph data object.
        transform (callable, optional): A function/transform that takes in the
            coarsened and pooled :obj:`torch_geometric.data.Data` object and
            returns a transformed version. (default: :obj:`None`)

    :rtype: :class:`torch_geometric.data.Data`
    """
    cluster, perm = consecutive_cluster(cluster)
    x = None if data.x is None else select(data.x, perm)
    index, attr = pool_edge(cluster, data.edge_index, data.edge_attr)
    batch = None if data.batch is None else pool_batch(perm, data.batch)
    pos = None if data.pos is None else pool_pos(cluster, data.pos)

    data = Batch(batch=batch, x=x, edge_index=index, edge_attr=attr, pos=pos)

    if transform is not None:
        data = transform(data)

    return data, perm


def max_pool_neighbor_x(
    data: Data,
    flow: Optional[str] = 'source_to_target',
) -> Data:
    r"""Max pools neighboring node features, where each feature in
    :obj:`data.x` is replaced by the feature value with the maximum value from
    the central node and its neighbors.
    """
    x, edge_index = data.x, data.edge_index

    edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)

    row, col = edge_index
    row, col = (row, col) if flow == 'source_to_target' else (col, row)

    data.x = scatter(x[row], col, dim=0, dim_size=data.num_nodes, reduce='max')
    return data


import math
import torch.nn.functional as F
import torch.nn as nn


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
    b, c, h, w = clst_feats.shape
    neighbor_range = candidate_radius * 2 + 1
    kernel_size = (stride[0]*neighbor_range, stride[1]*neighbor_range)
    padding = (stride[0]*candidate_radius, stride[1]*candidate_radius)
    n_candidate_pixels = kernel_size[0] * kernel_size[1]
    unfold_elem_feats = F.unfold(elem_feats, kernel_size, padding=padding, stride=stride)
    unfold_elem_feats = unfold_elem_feats.reshape(b, c, n_candidate_pixels, h, w)
    similarities = torch.einsum('bcphw,bchw->bphw', (unfold_elem_feats, clst_feats))
    similarities = torch.where(similarities==0, -torch.inf, similarities)
    if stable:
        similarities = similarities - similarities.max(1, keepdim=True).values.detach()
    soft_assignemnt = torch.softmax(similarities / tau, dim=1)
    new_clst_feats = torch.einsum('bphw,bcphw->bchw', (soft_assignemnt, unfold_elem_feats))
    return new_clst_feats, soft_assignemnt, similarities

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
    batch_size, channels, height, width = elem_feats.shape
    n_spixels = clst_feats.shape[2] * clst_feats.shape[3]
    neighbor_range = candidate_radius * 2 + 1
    candidate_clusters = F.unfold(clst_feats, kernel_size=neighbor_range, padding=candidate_radius)
    candidate_clusters = candidate_clusters.reshape(batch_size, channels, neighbor_range**2, n_spixels)
    unfold_elem_feats = F.unfold(elem_feats, kernel_size=stride, stride=stride)
    unfold_elem_feats = unfold_elem_feats.reshape(batch_size, channels, stride[0] * stride[1], n_spixels)
    similarities = torch.einsum('bkcn,bkpn->bcpn', (candidate_clusters, unfold_elem_feats))
    similarities = similarities.contiguous().reshape(batch_size * neighbor_range**2, -1, n_spixels)
    similarities = F.fold(similarities, (height, width), kernel_size=stride, stride=stride)
    similarities = similarities.reshape(batch_size, neighbor_range**2, height, width)
    # masking zero padding regions with -inf
    # by using the fact that the inner product is zero.
    similarities = torch.where(similarities==0, -torch.inf, similarities)
    if stable:
        similarities = similarities - similarities.max(1, keepdim=True).values.detach()
    soft_assignment = (similarities / tau).softmax(1)
    return soft_assignment, similarities

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
            clst_feats, s2p_assign, _ = update_clst_feats(x, clst_feats, stride, self.tau, self.candidate_radius)
            if self.normalize:
                clst_feats = clst_feats / clst_feats.norm(dim=1, keepdim=True)
        # compute a pixel-to-superpixel assignment
        p2s_assign, _ = compute_elem_to_center_assignment(clst_feats, x, stride, self.tau, self.candidate_radius)
        # remove the padding region
        if pad_y > 0:
            p2s_assign = p2s_assign[..., :-pad_y, :]
        if pad_x > 0:
            p2s_assign = p2s_assign[..., :-pad_x]
        return clst_feats, p2s_assign, s2p_assign

    def extra_repr(self):
        return f'n_spixels={self.n_spixels}, \n ' \
               f'n_iter={self.n_iter}, \n ' \
               f'tau={self.tau}, \n ' \
               f'candidate_radius={self.candidate_radius}, \n' 






# def TokenPooling(f, K):
    # f = B, N, D
    # First randomly initialize centers
f = torch.randn(10, 256, 32, 32)
K = 256
slic_fn = DiffSLIC(n_spixels=K, n_iter=5, tau=0.01, candidate_radius=1, stable=True)

# result = model(f)
# print(result.labels)

# rgb_img = torch.arange(30000).reshape(1, 3, 100, 100)
features, spix2pix_assign, pix2spix_assign = slic_fn(f)
print(features.size())
# iterations = 10
# indices = torch.randint(0, f.size(1), (f.size(0), K))
# centers = torch.gather(f, 1, indices.unsqueeze(-1).repeat(1, 1, f.size(-1)))
# for _ in range(iterations):
#     print(torch.norm(f.unsqueeze(2) - centers.unsqueeze(1), p=2, dim=3).size())
#     assert(0)

    