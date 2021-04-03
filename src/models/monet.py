""" Implementation of "Gaussian Mixture Model Convolutional Networks" (CVPR 17) """
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn import GMMConv
import torch.nn.functional as F
import torch_geometric.transforms as T
from src.utils.mapper import configmapper
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (graclus, max_pool, global_mean_pool)

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

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.relu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))

        x = global_mean_pool(data.x, data.batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)
