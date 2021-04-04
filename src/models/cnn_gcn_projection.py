"""Combine a custom CNN and GCN based on the config."""

from torch.nn import (
    BatchNorm2d,
    Conv2d,
    CrossEntropyLoss,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)
import torch

from src.utils.mapper import configmapper


@configmapper.map("models", "cnn_gcn_projection")
class CnnGcnProjection(Module):
    def __init__(self, config):
        super(CnnGcnProjection, self).__init__()
        self.cnn = configmapper.get_object("models",config.cnn_config.name)(config.cnn_config)
        self.gcn = configmapper.get_object("models",config.gcn_config.name)(config.gcn_config)

        self.linear_layer = Linear(config.cnn_config.num_classes+config.gcn_config.num_classes, config.num_classes)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, image, graph, labels=None):
        cnn_out = self.cnn(image)
        gcn_out = self.gcn(graph)
        out = torch.cat([cnn_out, gcn_out],dim=1)
        out = self.linear_layer(out)
        if labels is not None:
            loss = self.loss_fn(out, labels)
            return loss, out
        return out