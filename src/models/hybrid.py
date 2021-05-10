"""Combine a custom CNN and GCN based on the config."""

from torch.nn import Module
from torch.nn.functional import relu

from src.utils.mapper import configmapper


@configmapper.map("models", "hybrid")
class Hybrid(Module):
    def __init__(self, config):
        super().__init__()
        self.cnn = configmapper.get_object("models", config.cnn_config.name)(
            config.cnn_config
        )
        self.gcn = configmapper.get_object("models", config.gnn_config.name)(
            config.gnn_config
        )

    def forward(self, image, graph):
        cnn_out = self.cnn(image)
        gcn_out = self.gcn(graph)

        return (cnn_out, gcn_out)
