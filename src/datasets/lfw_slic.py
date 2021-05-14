"""Implements LFW SLIC Dataset"""
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_lfw_people
from torch.utils.data import Dataset
from torchvision import transforms

from src.modules.transforms import *
from src.utils.mapper import configmapper


@configmapper.map("datasets", "lfw_slic")
class LFWSlic(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

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

        self.data = fetch_lfw_people(data_home=config.filepath.data, color=True)

        if config.filepath.indices_csv != None:
            filtered_indices = list(pd.read_csv(config.filepath.indices_csv)["indices"])
            self.images = np.take(self.data.images, filtered_indices, axis=0)
            self.labels = np.take(self.data.target, filtered_indices, axis=0)
        else:
            self.images = self.data.images
            self.labels = self.data.target

        self.images = self.images.astype(np.uint8)

        # self.images = np.transpose(self.images, (0, 3, 1, 2))

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            graph = self.transform(image)

        return {"graph": graph, "label": torch.tensor(label, dtype=torch.long)}
