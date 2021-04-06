"""Implements TG MNIST SLIC Dataset"""
from torch.utils.data import Dataset
from torch_geometric import datasets
from torchvision import transforms

from src.modules.transforms import *
from src.utils.mapper import configmapper


@configmapper.map("datasets", "tg_mnist_slic")
class TgMnistSlic(Dataset):
    def __init__(self, config):
        self.config = config

        transformations = []
        if hasattr(config, "transform_args"):
            for transform in config.transform_args:
                param_dict = (
                    dict(transform["params"]) if transform["params"] is not None else {}
                )
                transformations.append(
                    configmapper.get_object("transforms", transform["type"])(
                        **param_dict
                    )
                )

        self.transform = (
            transforms.Compose(transformations) if transformations != [] else None
        )

        pre_transformations = []
        if hasattr(config, "pre_transform_args"):
            for pre_transform in config.pre_transform_args:
                param_dict = (
                    dict(pre_transform["params"])
                    if pre_transform["params"] is not None
                    else {}
                )
                pre_transformations.append(
                    configmapper.get_object("transforms", pre_transform["type"])(
                        **param_dict
                    )
                )

        self.pre_transform = (
            transforms.Compose(pre_transformations)
            if pre_transformations != []
            else None
        )
        self.dataset = datasets.MNISTSuperpixels(
            root=config.load_dataset_args.path,
            train=self.config.split == "train",
            transform=self.transform,
            pre_transform=self.pre_transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, example_idx):
        # essential to return as dict, hence the roundabout way of loading the dataset
        graph_data = self.dataset[example_idx]
        return {"graph": graph_data, "label": graph_data.y[0]}
