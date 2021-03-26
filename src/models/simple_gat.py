import torch
from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import dropout, relu
from torch_geometric.nn import GATConv, global_mean_pool

from src.utils.mapper import configmapper


@configmapper.map("models", "simple_gat")
class SimpleGAT(Module):
    def __init__(self, config):
        super(SimpleGAT, self).__init__()

        gat_hidden_layer_sizes = [
            config.num_node_features
        ] + list(config.gat_params.hidden_layer_sizes)
        linear_layer_sizes = (
            [gat_hidden_layer_sizes[-1]]
            + list(config.linear_layer_params.intermediate_layer_sizes)
            + [config.num_classes]
        )

        self.gatconv_layers = ModuleList(
            [
                GATConv(in_channels=in_channels, out_channels=out_channels, heads=1)
                for in_channels, out_channels in zip(
                    gat_hidden_layer_sizes[:-1], gat_hidden_layer_sizes[1:]
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
        # print(self.gatconv_layers)

    def forward(self, data):
        out, edge_index = data.x, data.edge_index
        # out = torch.cat([data.pos, data.x], dim=1)
        # print(edge_index.shape)
        # print(out.shape)

        for gatconv_layer in self.gatconv_layers:
            # print(out.shape)
            out = gatconv_layer(out, edge_index)

        for linear_layer in self.linear_layers:
            out = linear_layer(out)
            out = relu(out)

        return out
