import torch
from torch.nn import CrossEntropyLoss, Linear, Module, ModuleList
from torch.nn.functional import dropout, relu
from torch_geometric.nn import GATConv, global_mean_pool

from src.utils.mapper import configmapper


@configmapper.map("models", "simple_gat")
class SimpleGAT(Module):
    def __init__(self, config):
        super(SimpleGAT, self).__init__()
        gat_hidden_layer_sizes = [config.num_node_features] + list(
            config.gat_params.hidden_layer_sizes
        )
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

        self.loss_fn = CrossEntropyLoss()

    def forward(self, graph, labels=None):
        out, edge_index, batch = graph.x, graph.edge_index, graph.batch
        out = torch.cat([graph.pos, graph.x], dim=1)

        for gatconv_layer in self.gatconv_layers:
            out = gatconv_layer(out, edge_index)
            out = relu(out)

        out = global_mean_pool(out, batch)
        out = dropout(out, p=0.2, training=self.training)
        for linear_layer in self.linear_layers[:-1]:
            out = linear_layer(out)
            out = relu(out)

        out = self.linear_layers[-1](out)
        if labels is not None:
            loss = self.loss_fn(out, labels)
            return loss, out
        return out
