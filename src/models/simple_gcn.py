from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Module, Linear
from torch.nn.functional import relu, dropout

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
        out = self.conv1(out, edge_index)
        out = relu(out)
        out = self.conv2(out, edge_index)
        out = relu(out)
        out = self.conv3(out, edge_index)
        out = relu(out)

        out = global_mean_pool(out, batch)
        out = dropout(out, p=0.5, training=self.training)
        out = self.lin(out)

        return out
