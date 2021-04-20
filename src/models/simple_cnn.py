"""Implementation of a custom CNN with random weights."""

from torch.nn import (
    BatchNorm2d,
    Conv2d,
    CrossEntropyLoss,
    Linear,
    MaxPool2d,
    Module,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.nn.functional import avg_pool2d, dropout, relu

from src.utils.mapper import configmapper


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_channels),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


@configmapper.map("models", "simple_cnn")
class SimpleCnn(Module):
    def __init__(self, config):
        super(SimpleCnn, self).__init__()
        cnn_hidden_layer_sizes = [config.num_input_channels] + list(
            config.cnn_params.hidden_layer_sizes
        )

        linear_layer_sizes = (
            [cnn_hidden_layer_sizes[-1]]
            + list(config.linear_layer_params.intermediate_layer_sizes)
            + [config.num_classes]
        )

        self.cnn_layers = ModuleList(
            [
                ConvBlock(in_channels=in_channels, out_channels=out_channels)
                for in_channels, out_channels in zip(
                    cnn_hidden_layer_sizes[:-1], cnn_hidden_layer_sizes[1:]
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

    def forward(self, image, labels=None):
        out = image
        for cnn_layer in self.cnn_layers:
            out = cnn_layer(out)
        # out = out.view(out.size(0), -1)
        # Global Mean Pooling
        out = avg_pool2d(out, kernel_size=out.size()[2:]).view(out.size()[0], -1)
        out = dropout(out, p=0.2, training=self.training)
        for linear_layer in self.linear_layers[:-1]:
            out = linear_layer(out)
            out = relu(out)

        out = self.linear_layers[-1](out)
        if labels is not None:
            loss = self.loss_fn(out, labels)
            return loss, out
        return out
