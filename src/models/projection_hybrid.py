"""Combine a custom CNN and GCN based on the config."""

import torch
from torch.nn import CrossEntropyLoss, Linear, Module
from torch.nn.functional import relu

from src.utils.mapper import configmapper


@configmapper.map("models", "projection_hybrid")
class ProjectionHybrid(Module):
    def __init__(self, config):
        super(Projection, self).__init__()
        self.cnn = configmapper.get_object("models", config.cnn_config.name)(
            config.cnn_config
        )
        self.gcn = configmapper.get_object("models", config.gnn_config.name)(
            config.gnn_config
        )

    def forward(self, image, graph, labels=None):
        cnn_out = self.cnn(image)
        gcn_out = self.gcn(graph)

        return (cnn_out, gcn_out)
