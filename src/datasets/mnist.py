"""Implements MNIST Dataset"""
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from src.modules.transforms import *
from src.utils.mapper import configmapper


@configmapper.map("datasets", "mnist")
class Mnist(Dataset):
    def __init__(self, config):
        self.config = config

        transformations = []
        for transform in config.transform_args:
            param_dict = (
                dict(transform["params"]) if transform["params"] is not None else {}
            )
            if transform['type'] == 'Resize':
                param_dict['size'] = tuple(param_dict['size'])
            transformations.append(
                configmapper.get_object("transforms", transform["type"])(**param_dict)
            )
        self.transform = (
            transforms.Compose(transformations) if transformations != [] else None
        )

        self.dataset = datasets.MNIST(
            config.load_dataset_args.path,
            download=True,
            train=self.config.split == "train",
            transform=self.transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, example_idx):
        # essential to return as dict, hence the roundabout way of loading the dataset
        img, label = self.dataset[example_idx]
        return {"image": img, "labels": label}
