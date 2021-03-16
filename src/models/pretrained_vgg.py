"""Implementation of a CNN for classfication with VGG Encoder."""

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

import torchvision.models as models

from src.utils.mapper import configmapper

STR_MODEL_MAPPING = {
    "11": models.vgg11,
    "13": models.vgg13,
    "16": models.vgg16,
    "19": models.vgg19,
    "11_bn": models.vgg11_bn,
    "13_bn": models.vgg13_bn,
    "16_bn": models.vgg16_bn,
    "19_bn": models.vgg19_bn,
}


@configmapper.map("models", "pretrained_vgg")
class PretrainedVGG(Module):
    def __init__(self, config):
        super(PretrainedVGG, self).__init__()
        self.config = config

        vgg_version = config.vgg_version
        if config.batch_norm:
            vgg_version += "_" + "bn"

        assert vgg_version in STR_MODEL_MAPPING, "VGG Version incorrect."
        self.pretrained_vgg = STR_MODEL_MAPPING[vgg_version](pretrained=True)

        # modify the last linear layer
        in_features_dim = self.pretrained_vgg.classifier[-1].in_features
        self.pretrained_vgg.classifier[-1] = Linear(in_features_dim, config.num_classes)

        # freeze the encoder
        # can add functionality later to freeze a specified number of layers
        if config.freeze_encoder:
            for param in self.pretrained_vgg.features.parameters():
                param.requires_grad = False

        self.loss_fn = CrossEntropyLoss()

    def forward(self, image, labels=None):
        if image.shape[1] == 1:
            image = torch.cat((image, image, image), dim=1)
        logits = self.pretrained_vgg(image)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, out
        return out
