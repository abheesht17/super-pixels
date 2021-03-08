import numpy as np
from src.utils.mapper import configmapper
from torch.nn import BatchNorm2d, Conv2d, Linear, MaxPool2d, Module, ReLU, Sequential


@configmapper.map("models", "simple_cnn")
class SimpleCnn(Module):
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
