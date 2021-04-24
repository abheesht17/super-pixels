import torch
from torch.nn import CrossEntropyLoss, Linear, Module, ModuleList
from torch.nn.functional import dropout, relu
from torch_geometric.nn import GatConv, global_mean_pool

from src.utils.mapper import configmapper


@configmapper.map("models", "multigat")
class MultiGat(Module):
    def __init__(self, config):
        super(MultiGat, self).__init__()

        num_heads = config.gat_params.num_heads
        gat_hidden_layer_sizes = list(config.gat_params.hidden_layer_sizes)
        # gat_hidden_layer_sizes = [layer_size*num_head for layer_size,num_head in zip(gat_hidden_layer_sizes,num_heads)]

        gat_hidden_layer_sizes = [config.num_node_features] + gat_hidden_layer_sizes

        linear_layer_sizes = (
            [gat_hidden_layer_sizes[-1] * num_heads[-1]]
            + list(config.linear_layer_params.intermediate_layer_sizes)
            + [config.num_classes]
        )

        self.gatconv_layers = ModuleList(
            [
                GatConv(
                    in_channels=in_channels * head_mul,
                    out_channels=out_channels,
                    heads=heads,
                )
                for in_channels, out_channels, heads, head_mul in zip(
                    gat_hidden_layer_sizes[:-1],
                    gat_hidden_layer_sizes[1:],
                    num_heads,
                    [1] + num_heads[1:],
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

        out = global_mean_pool(out, batch)
        out = dropout(out, p=0.2, training=self.training)
        for linear_layer in self.linear_layers:
            out = linear_layer(out)
            out = relu(out)
        if labels is not None:
            loss = self.loss_fn(out, labels)
            return loss, out

        return out
