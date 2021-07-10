"""Implementation of an MLP."""

from torch.nn import CrossEntropyLoss, Linear, Module, ModuleList, ReLU, Sequential
from torch.nn.functional import dropout, relu

from src.utils.mapper import configmapper


class LinearBlock(Module):
    def __init__(self, in_features, out_features, if_relu=True):
        super(ConvBlock, self).__init__()

        if if_relu:
            self.block = Sequential(
                Linear(in_features, out_features),
                ReLU(),
            )
        else:
            self.block = Sequential(
                Linear(in_features, out_features),
            )

    def forward(self, x):
        return self.block(x)


@configmapper.map("models", "mlp")
class SimpleMLP(Module):
    def __init__(self, config):
        super(SimpleMLP, self).__init__()
        linear_layer_sizes = (
            [config.num_input_features]
            + list(config.mlp_params.hidden_layer_sizes)
            + [config.num_classes]
        )

        if_relus = [True for i in range(len(linear_layer_sizes) - 1)]
        if_relus[-1] = False
        self.mlp_layers = ModuleList(
            [
                LinearBlock(
                    in_features=in_features, out_features=out_features, if_relu=if_relu
                )
                for in_features, out_features, if_relu in zip(
                    linear_layer_sizes[:-1], linear_layer_sizes[1:], if_relus
                )
            ]
        )

        self.loss_fn = CrossEntropyLoss()

    def forward(self, image, labels=None):
        out = torch.flatten(image, start_dim=1, end_dim=2)

        for mlp_layer in self.mlp_layers:
            out = mlp_layer(out)

        if labels is not None:
            loss = self.loss_fn(out, labels)
            return loss, out
        return out
