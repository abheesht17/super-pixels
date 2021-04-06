import torch
from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import dropout, relu
from torch_geometric.nn import GCNConv, global_mean_pool

from src.utils.mapper import configmapper

@configmapper.map("models", "simple_gcn")
class SimpleGcn(Module):
    def __init__(self, config):
        super(SimpleGcn, self).__init__()

        gcn_hidden_layer_sizes = [config.num_node_features] + list(
            config.gcn_params.hidden_layer_sizes
        )
        linear_layer_sizes = (
            [gcn_hidden_layer_sizes[-1]]
            + list(config.linear_layer_params.intermediate_layer_sizes)
            + [config.num_classes]
        )

        self.gcnconv_layers = ModuleList(
            [
                GCNConv(in_channels=in_channels,
                        out_channels=out_channels)
                for in_channels, out_channels in zip(
                    gcn_hidden_layer_sizes[:-1], gcn_hidden_layer_sizes[1:]
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

    def forward(self, data):
        out, edge_index, batch = data.x, data.edge_index, data.batch
        out = torch.cat([data.pos, data.x], dim=1)
        
        for gcnconv_layer in self.gcnconv_layers:
            out = gcnconv_layer(out, edge_index)
            out = relu(out)

        out = global_mean_pool(out, batch)
        out = dropout(out, p=0.2, training=self.training)
        
        for linear_layer in self.linear_layers[:-1]:
            out = linear_layer(out)
            out = relu(out)

        out = self.linear_layers[-1](out)
        return out
