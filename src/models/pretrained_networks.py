"""Commonly used Pretrained Networks"""
import torch
import torchvision.models as models
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


@configmapper.map("models", "vgg")
class VGG(Module):
    """

    Attributes:
        vgg16 (torch.nn.Sequential): Pretrained VGG-16 Model
        linear_layers (torch.nn.Linear): Linear layer for classification.

    Methods:
        forward

    """

    def __init__(self, config):
        super(VGG, self).__init__()
        vgg_variants = [11, 13, 16, 19]
        print(config)
        assert config.variant in vgg_variants, "VGG only has variants " + str(
            vgg_variants
        )
        if config.variant == 11:
            self.vgg = models.vgg11(pretrained=True)
        elif config.variant == 13:
            self.vgg = models.vgg13(pretrained=True)
        elif config.variant == 16:
            self.vgg = models.vgg16(pretrained=True)
        elif config.variant == 19:
            self.vgg = models.vgg19(pretrained=True)
        self.linear_layers = Linear(1000, config.num_labels)
        self.loss_fn = CrossEntropyLoss()
        self.grayscale = config.grayscale

    def forward(self, image, labels=None):
        if self.grayscale:
            # (batchsize, 1, img_size, img_size) -> (batchsize, 3, img_size, img_size)
            image = torch.cat((image,) * 3, axis=1)

        x_out = self.vgg(image)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.linear_layers(x_out)
        if labels is not None:
            loss = self.loss_fn(x_out, labels)
            return loss, x_out
        return x_out
