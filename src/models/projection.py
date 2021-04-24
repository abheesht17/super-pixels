"""Combine a custom CNN and GCN based on the config."""

import torch
from torch.nn import CrossEntropyLoss, Linear, Module
from torch.nn.functional import relu

from src.utils.mapper import configmapper


@configmapper.map("models", "projection")
class Projection(Module):
    def __init__(self, config):
        super(Projection, self).__init__()
        self.cnn = configmapper.get_object("models", config.cnn_config.name)(
            config.cnn_config
        )
        self.gcn = configmapper.get_object("models", config.gcn_config.name)(
            config.gcn_config
        )

        self.linear_layer = Linear(
            config.cnn_config.num_classes + config.gcn_config.num_classes,
            config.num_classes,
        )
        self.loss_fn = CrossEntropyLoss()

    def forward(self, image, graph, labels=None):
        cnn_out = self.cnn(image)
        gcn_out = self.gcn(graph)
        out = torch.cat([cnn_out, gcn_out], dim=1)
        out = relu(out)
        out = self.linear_layer(out)
        if labels is not None:
            loss = self.loss_fn(out, labels)
            return loss, out
        return out
