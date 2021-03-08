import numpy as np
from src.utils.mapper import configmapper
from torch.nn import BatchNorm2d, Conv2d, Linear, MaxPool2d, Module, ReLU, Sequential


@configmapper.map("models", "simple_cnn")
class SimpleCnn(Module):
    def __init__(self, config):
        super(SimpleCnn, self).__init__()
        self.cnn_layers = [
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        ]
        self.linear_layers = Linear(4 * 7 * 7, 10)

    def forward(self, x_in):
        x_out = x_in["image"]
        for layer in self.cnn_layers:
            x_out = layer(x_out)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.linear_layers(x_out)
        return x_out
