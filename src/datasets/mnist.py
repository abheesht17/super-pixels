import torch
from src.modules.transforms import *
from src.utils.mapper import configmapper
from torchvision import transforms

from datasets import load_dataset


@configmapper.map("datasets", "mnist")
class Mnist:
    def __init__(self, config):
        self.config = config
        self.raw_dataset = load_dataset(config.load_dataset_args.path)
        self.raw_dataset.set_format(
            "numpy",
            columns=["image"],
            output_all_columns=True,
        )
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
        self.train_dataset = self.raw_dataset.map(
            self.prepare_train_features, batched=True, batch_size=10000
        )

        self.train_dataset.set_format(
            "torch", columns=["image", "label"], dtype=torch.float32
        )

    def prepare_train_features(self, examples):
        if self.transform is not None:
            for example_idx, example in enumerate(examples["image"]):
                examples["image"][example_idx] = self.transform(
                    examples["image"][example_idx]
                )
        return examples
