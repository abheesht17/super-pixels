"""Implementation of a CNN for classfication with VGG Encoder."""

import torch
import torchvision.models as models
from torch.nn import CrossEntropyLoss, Linear, Module

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

        vgg_version = config.vgg_version
        num_layers = int(vgg_version)

        # check that the config.num_layers_freeze < number of layers in the network
        num_layers_freeze = config.num_layers_freeze
        assert num_layers_freeze < int(vgg_version), (
            "(num_layers_freeze) should be greater than (number of layers in network - 1). num_layers_freeze = "
            + str(num_layers_freeze)
            + " and number of layers in network = "
            + vgg_version
        )

        if config.batch_norm:
            vgg_version += "_" + "bn"
        assert (
            vgg_version in STR_MODEL_MAPPING
        ), 'VGG version incorrect, should be in ["11","13","16","19"]'

        # load the pretrained model
        self.pretrained_vgg = STR_MODEL_MAPPING[vgg_version](
            pretrained=config.pretrained
        )

        # freeze specified number of layers
        num_cls_layers_freeze = 3 - num_layers + num_layers_freeze
        if num_cls_layers_freeze > 0:
            num_enc_layers_freeze = num_layers_freeze - num_cls_layers_freeze
        else:
            num_cls_layers_freeze = 0
            num_enc_layers_freeze = num_layers_freeze

        if config.batch_norm:
            self.pretrained_vgg.features = self.freeze_layers(
                self.pretrained_vgg.features, num_enc_layers_freeze, 4
            )
        else:
            self.pretrained_vgg.features = self.freeze_layers(
                self.pretrained_vgg.features, num_enc_layers_freeze, 2
            )

        self.pretrained_vgg.classifier = self.freeze_layers(
            self.pretrained_vgg.classifier, num_cls_layers_freeze, 2
        )

        # modify the last linear layer
        in_features_dim = self.pretrained_vgg.classifier[-1].in_features
        self.pretrained_vgg.classifier[-1] = Linear(in_features_dim, config.num_classes)

        self.loss_fn = CrossEntropyLoss()

    def freeze_layers(self, model, num_layers_freeze_param, mod):
        ct_unique = 0
        k = 0
        for name, param in model.named_parameters():
            if k % mod == 0:
                ct_unique += 1
            if param.requires_grad and ct_unique <= num_layers_freeze_param:
                param.requires_grad = False
            k += 1
        return model

    def forward(self, image, labels=None):

        logits = self.pretrained_vgg(image)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits
