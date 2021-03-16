"""Implements MNIST Dataset"""
import numpy as np
from torchvision import transforms

import datasets
from datasets import DatasetDict, load_dataset
from src.modules.transforms import *
from src.utils.mapper import configmapper


@configmapper.map("datasets", "hf_image_classification")
class HFImageClassification:
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

        features = datasets.Features(
            {
                self.image_column_name: datasets.Array3D(
                    shape=tuple(self.config.features.image_output_shape),
                    dtype="float32",
                ),
                self.label_column_name: datasets.features.ClassLabel(
                    names=list(self.config.features.label_names)
                ),
            }
        )

        self.train_dataset = self.raw_dataset.map(
            self.prepare_features,
            features=features,
            batched=True,
            batch_size=1000,
        )

        if self.image_column_name != "image":
            self.train_dataset = self.train_dataset.rename_column(
                self.image_column_name, "image"
            )
        if self.label_column_name != "label":
            self.train_dataset = self.train_dataset.rename_column(
                self.label_column_name, "label"
            )

        self.train_dataset.set_format("torch", columns=["image", "label"])

    def prepare_features(self, examples):
        images = []
        labels = []
        for example_idx, example in enumerate(examples[self.image_column_name]):
            if self.channels_first_input:
                if self.transform is not None:
                    images.append(
                        self.transform(examples[self.image_column_name][example_idx])
                    )
                else:
                    images.append(examples[self.image_column_name][example_idx])
            else:
                if self.transform is not None:
                    images.append(
                        self.transform(
                            examples[self.image_column_name][example_idx].permute(
                                2, 0, 1
                            )
                        )
                    )
                else:
                    images.append(
                        examples[self.image_column_name][example_idx].permute(2, 0, 1)
                    )

            labels.append(examples[self.label_column_name][example_idx])
        output = {self.label_column_name: labels, self.image_column_name: images}
        return output
