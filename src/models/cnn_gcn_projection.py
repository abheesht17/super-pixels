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

from src.utils.mapper import configmapper


@configmapper.map("models", "cnn_gcn_projection")
class CnnGcnProjection(Module):
    def __init__(self, config):
        super(CnnGcnProjection, self).__init__()
        self.cnn = configmapper.get_object(config.cnn_config.name)(config.cnn_config)
        self.gcn = configmapper.get_object(config.gcn_config.name)(config.gcn_config)

        self.loss_fn = CrossEntropyLoss()

    def forward(self, image, graph labels=None):
        cnn_out = self.cnn(image)
        gcn_out = out.view(out.size(0), -1)
        out = self.linear_layers(out)
        if labels is not None:
            loss = self.loss_fn(out, labels)
            return loss, out
        return out
