""" Implementation of "Gaussian Mixture Model Convolutional Networks" (CVPR 17) """
import math

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Parameter
from torch_geometric.nn import GMMConv, global_mean_pool, graclus, max_pool
from torch_geometric.utils import normalized_cut

from src.utils.mapper import configmapper


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


@configmapper.map("models", "monet")
class MoNet(torch.nn.Module):
    def __init__(self, config):
        super(MoNet, self).__init__()
        self.conv1 = GMMConv(1, 32, dim=2, kernel_size=config.kernel_size)
        self.conv2 = GMMConv(32, 64, dim=2, kernel_size=config.kernel_size)
        self.conv3 = GMMConv(64, 64, dim=2, kernel_size=config.kernel_size)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, graph):
        graph.x = F.elu(self.conv1(graph.x, graph.edge_index, graph.edge_attr))
        weight = normalized_cut_2d(graph.edge_index, graph.pos)
        cluster = graclus(graph.edge_index, weight, graph.x.size(0))
        graph.edge_attr = None
        graph = max_pool(cluster, graph, transform=T.Cartesian(cat=False))

        graph.x = F.relu(self.conv2(graph.x, graph.edge_index, graph.edge_attr))
        weight = normalized_cut_2d(graph.edge_index, graph.pos)
        cluster = graclus(graph.edge_index, weight, graph.x.size(0))
        graph = max_pool(cluster, graph, transform=T.Cartesian(cat=False))

        graph.x = F.elu(self.conv3(graph.x, graph.edge_index, graph.edge_attr))

        x = global_mean_pool(graph.x, graph.batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)
