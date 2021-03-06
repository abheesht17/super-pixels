from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear


class SimpleCnn(Module):
    def __init__(self):
        super(SimpleCnn, self).__init__()
        self.cnn_layers = Sequential(
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = Sequential(Linear(4 * 7 * 7, 10))

    def forward(self, x_in):
        x_out = self.cnn_layers(x_in)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.linear_layers(x_out)
        return x_out
