"""Commonly used Pretrained Networks"""
import torch
import torchvision.models as models
from torch.nn import BatchNorm2d, Conv2d, Linear, MaxPool2d, Module, ReLU, Sequential, CrossEntropyLoss
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
        # if config.grayscale: 
        #     self.vgg16 = models.vgg16(pretrained=True)
        #     self.vgg16.features[0] = Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # else:
        #     
        self.vgg16 = models.vgg16(pretrained=True)
        self.linear_layers = Linear(1000, config.num_labels)
        self.loss_fn = CrossEntropyLoss()
        self.grayscale = config.grayscale

    def forward(self, image, labels=None):
        if self.grayscale:
            # (batchsize, 1, img_size, img_size) -> (batchsize, 3, img_size, img_size) 
            image = torch.cat((image,)*3, axis=1)

        x_out = self.vgg16(image)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.linear_layers(x_out)
        if labels is not None:
            loss = self.loss_fn(x_out, labels)
            return loss, out
        return x_out
