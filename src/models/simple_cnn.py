"""Implementation of a custom CNN with random weights."""

from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
    CrossEntropyLoss,
)

from src.utils.mapper import configmapper


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

        self.loss_fn = CrossEntropyLoss()

    def forward(self, image, labels=None):
        out = self.cnn_layers(image)
        out = out.view(out.size(0), -1)
        out = self.linear_layers(out)
        if labels is not None:
            loss = self.loss_fn(out, labels)
            return loss, out
        return out
