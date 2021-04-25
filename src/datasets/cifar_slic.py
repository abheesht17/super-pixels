"""Implements CIFAR SLIC Dataset"""
import pickle
import struct

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.modules.transforms import *
from src.utils.mapper import configmapper
from src.utils.viz import visualize_geometric_graph


@configmapper.map("datasets", "cifar_slic")
class CifarSlic(Dataset):
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

        with open(config.filepath.data, "rb") as f:

            self.data = pickle.load(f, encoding="bytes")
        self.images = self.data[b"data"]
        self.labels = self.data[config.label.encode("UTF-8")]

        if config.filepath.indices_csv != None:
            filtered_indices = list(pd.read_csv(config.filepath.indices_csv)["index"])
            self.images = np.take(self.images, filtered_indices, axis=0)
            self.labels = np.take(self.labels, filtered_indices, axis=0)

        self.images = np.transpose(
            np.reshape(self.images, (-1, 3, 32, 32)), (0, 2, 3, 1)
        )

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        print(image.shape)
        if self.transform is not None:
            graph = self.transform(image)
        plt.imsave(f"{label}.png", image)
        visualize_geometric_graph(graph, "{label}_graph.png")

        return {"graph": graph, "label": torch.tensor(label, dtype=torch.long)}
