"""Implements LFW Img Slic sDataset"""
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.modules.transforms import *
from src.utils.mapper import configmapper


@configmapper.map("datasets", "lfw_img_slic")
class LFWImgSlic(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        graph_transformations = []
        for transform in config.graph_transform_args:
            param_dict = (
                dict(transform["params"]) if transform["params"] is not None else {}
            )
            graph_transformations.append(
                configmapper.get_object("transforms", transform["type"])(**param_dict)
            )
        self.graph_transform = (
            transforms.Compose(graph_transformations)
            if graph_transformations != []
            else None
        )

        image_transformations = []
        for transform in config.image_transform_args:
            param_dict = (
                dict(transform["params"]) if transform["params"] is not None else {}
            )
            image_transformations.append(
                configmapper.get_object("transforms", transform["type"])(**param_dict)
            )
        self.image_transform = (
            transforms.Compose(image_transformations)
            if image_transformations != []
            else None
        )

        self.data = fetch_lfw_people(data_home=config.filepath.data, color=True)

        if config.filepath.indices_csv != None:
            filtered_indices = list(pd.read_csv(config.filepath.indices_csv)["indices"])
            self.images = np.take(self.data.images, filtered_indices, axis=0)
            self.labels = np.take(self.data.target, filtered_indices, axis=0)

        self.images = np.transpose(self.images, (0, 3, 1, 2))

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.graph_transform is not None:
            graph = self.graph_transform(image)

        if self.image_transform is not None:
            image = self.image_transform(image)

        return {
            "graph": graph,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }
