"""Implements MNIST Dataset"""
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

import datasets
from datasets import DatasetDict, load_dataset
from src.modules.transforms import *
from src.utils.mapper import configmapper


@configmapper.map("datasets", "hf_graph_classification")
class HFGraphClassification:
    def __init__(self, config):
        self.config = config
        self.image_column_name = config.image_column_name
        self.label_column_name = config.label_column_name
        self.channels_first_input = config.channels_first_input

        transformations = []
        for transform in config.transform_args:
            param_dict = (
                dict(transform["params"]) if transform["params"] is not None else {}
            )
            transformations.append(
                configmapper.get_object("transforms", transform["type"])(**param_dict)
            )
        self.transform = (
            transforms.Compose(transformations) if transformations != [] else None
        )

        self.raw_dataset = load_dataset(**config.load_dataset_args)
        if config.remove_columns is not None:
            self.raw_dataset = self.raw_dataset.remove_columns(config.remove_columns)
        self.raw_dataset.set_format(
            "torch", columns=self.raw_dataset["train"].column_names
        )

        t = self.transform(self.raw_dataset["train"]["image"][0])
        print(t)
        print(len(t.x))
        print(t.shape)
