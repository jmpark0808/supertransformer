from typing import Callable, List, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, TopKPooling, graclus#, max_pool
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.utils.repeat import repeat
from torch_geometric.utils import normalized_cut
import torch_geometric.transforms as T
from Blocks.GraphPool import max_pool
from torch_geometric.utils import dropout_edge

transform = T.Cartesian(cat=False)
def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))





class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth_pool: int,
        depth_conv: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
        dropout: float = 0, 
        heads: int = 8,
        dropout_edge: float = 0, 
    ):
        super().__init__()
        assert depth_pool >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth_pool = depth_pool
        self.pool_ratios = repeat(pool_ratios, depth_pool)
        self.act = activation_resolver(act)
        self.sum_res = sum_res
        self.dropout_edge = dropout_edge

        channels = hidden_channels

        self.convs = nn.ModuleList([GATv2Conv(in_channels=channels, out_channels=channels,
                                                                          heads=heads, dropout=dropout, edge_dim=None,
                                                                            concat=False) for _ in range(depth_conv-1)])
    
        self.ln1s = nn.ModuleList([GraphNorm(channels) for _ in range(depth_conv-1)])

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.down_convs.append(GATv2Conv(in_channels, channels, heads=heads, concat=False, dropout=dropout, edge_dim=None))
        self.norms.append(GraphNorm(channels))
        for i in range(depth_pool):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GATv2Conv(channels, channels, heads=heads, concat=False, dropout=dropout, edge_dim=None))
            self.norms.append(GraphNorm(channels))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        self.up_norms = torch.nn.ModuleList()
        for i in range(depth_pool - 1):
            self.up_convs.append(GATv2Conv(in_channels, channels, heads=heads, concat=False, dropout=dropout, edge_dim=None))
            self.up_norms.append(GraphNorm(channels))
        self.up_convs.append(GATv2Conv(in_channels, out_channels, heads=heads, concat=False, dropout=dropout, edge_dim=None))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()


    def forward(self, data,
                batch: OptTensor = None) -> Tensor:
        """"""  # noqa: D419
        if batch is None:
            batch = edge_index.new_zeros(data.x.size(0))

        edge_index_, edge_mask = dropout_edge(data.edge_index, p=self.dropout_edge, training=self.training)
        

        data.x = self.down_convs[0](data.x, edge_index_)
        data.x = self.norms[0](data.x, batch=batch)
        data.x = self.act(data.x)

        for ln, conv in zip(self.ln1s, self.convs):
            h = data.x
            edge_index_, edge_mask = dropout_edge(data.edge_index, p=self.dropout_edge, training=self.training)
            data.x = conv(data.x, edge_index=edge_index_, edge_attr=None)
            data.x = ln(data.x, batch=batch)
            data.x = self.act(data.x) + h

        xs = [data.x]
        edge_indices = [data.edge_index]
        batches = [batch]
        perms = []
        

        for i in range(1, self.depth_pool + 1):
            # edge_weight = data.x.new_ones(data.edge_index.size(1))
            # data.edge_index, edge_weight = self.augment_adj(data.edge_index, edge_weight,
            #                                            data.x.size(0))
            
         
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            data.edge_attr = None
            data, perm = max_pool(cluster, data, transform=transform)
            # x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
            #     x, edge_index, edge_weight, batch)

            edge_index_, edge_mask = dropout_edge(data.edge_index, p=self.dropout_edge, training=self.training)
            data.x = self.down_convs[i](data.x, edge_index_)
            data.x = self.norms[i](data.x, batch=data.batch)
            data.x = self.act(data.x)

            if i < self.depth_pool:
                xs += [data.x]
                batches += [data.batch]
            edge_indices += [data.edge_index]
            perms += [perm]

        for i in range(self.depth_pool):
            j = self.depth_pool - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            perm = perms[j]
            batch = batches[j]

            up = torch.zeros_like(res)
            up[perm] = data.x
            data.x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            data.x = self.up_convs[i](data.x, edge_index)
            data.x = self.up_norms[i](data.x, batch=batch) if i < self.depth_pool - 1 else data.x
            data.x = self.act(data.x) if i < self.depth_pool - 1 else data.x

        return data.x, perms, edge_indices


    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight,
                                  size=(num_nodes, num_nodes))
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        # edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth_pool}, pool_ratios={self.pool_ratios})')