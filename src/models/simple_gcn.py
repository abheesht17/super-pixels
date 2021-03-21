import torch
from torch.nn import Linear, Module
from torch.nn.functional import dropout, relu
from torch_geometric.nn import GCNConv, global_mean_pool

from src.utils.mapper import configmapper


@configmapper.map("models", "simple_gcn")
class SimpleGcn(Module):
    def __init__(self, config):
        super(SimpleGcn, self).__init__()
        self.conv1 = GCNConv(config.num_node_features, config.hidden_channels)
        self.conv2 = GCNConv(config.hidden_channels, config.hidden_channels)
        self.conv3 = GCNConv(config.hidden_channels, config.hidden_channels)
        self.lin = Linear(config.hidden_channels, config.num_classes)

    def forward(self, data):
        out, edge_index, batch = data.x, data.edge_index, data.batch
        out = torch.cat([data.pos, data.x], dim=1)
        out = self.conv1(out, edge_index)
        out = relu(out)
        out = self.conv2(out, edge_index)
        out = relu(out)
        out = self.conv3(out, edge_index)
        out = relu(out)

        out = global_mean_pool(out, batch)
        out = dropout(out, p=0.2, training=self.training)
        out = self.lin(out)

        return out
