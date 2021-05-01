""" Implementation of "Gaussian Mixture Model Convolutional Networks" (CVPR 17) """
import math

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear, Module, ModuleList, Parameter, ReLU, Sequential
from torch_geometric.nn import GMMConv, global_mean_pool, graclus, max_pool
from torch_geometric.utils import normalized_cut

from src.utils.mapper import configmapper


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


@configmapper.map("models", "monet")
class MoNet(Module):
    def __init__(self, config):
        super(MoNet, self).__init__()
        monet_hidden_layer_sizes = [config.num_node_features] + list(
            config.monet_params.hidden_layer_sizes
        )
        linear_layer_sizes = (
            [monet_hidden_layer_sizes[-1]]
            + list(config.linear_layer_params.intermediate_layer_sizes)
            + [config.num_classes]
        )
        self.monet_layers = ModuleList(
            [
                GMMConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dim=2,
                    kernel_size=config.kernel_size,
                )
                for in_channels, out_channels in zip(
                    monet_hidden_layer_sizes[:-1], monet_hidden_layer_sizes[1:]
                )
            ]
        )
        self.linear_layers = ModuleList(
            [
                Linear(in_features=in_features, out_features=out_features)
                for in_features, out_features in zip(
                    linear_layer_sizes[:-1], linear_layer_sizes[1:]
                )
            ]
        )

    def forward(self, graph):
        data = graph
        data.x = torch.cat([data.pos, data.x], dim=1)
        for i, monet_layer in enumerate(self.monet_layers[:-1]):
            data.x = F.relu(monet_layer(data.x, data.edge_index, data.edge_attr))
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            if i == 0:
                data.edge_attr = None  # Check what this does!!!
                data.edge_attr = None 
            data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
            
        data.x = self.monet_layers[-1](data.x, data.edge_index, data.edge_attr)

        for linear_layer in self.linear_layers[:-1]:
            x = global_mean_pool(data.x, data.batch)
            x = F.relu(linear_layer(x))
            x = F.dropout(x)

        return F.log_softmax(self.linear_layers[-1](x), dim=1)
