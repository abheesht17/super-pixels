"""Implementation of a custom CNN with random weights."""

from torch.nn import (BatchNorm2d, Conv2d, Linear, MaxPool2d, Module, ReLU,
                      Sequential)

from src.utils.mapper import configmapper


@configmapper.map("models", "simple_cnn")
class SimpleCnn(Module):
    """

    Attributes:
        cnn_layers (torch.nn.Sequential): Sequential object containing the convolutional layers.
        linear_layers (torch.nn.Linear): Linear layer for classification.

    Methods:
        forward

    """

    def __init__(self, config):
        super(SimpleCnn, self).__init__()
        self.cnn_layers = Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = Linear(32 * 3 * 3, 10)

    def forward(self, x_in):
        x_out = self.cnn_layers(x_in["image"])
        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.linear_layers(x_out)
        return x_out
