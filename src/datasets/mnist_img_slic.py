"""Implements MNIST Dataset"""
import struct
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from src.modules.transforms import *
from src.utils.mapper import configmapper
from src.utils.viz import visualize_geometric_graph


@configmapper.map("datasets", "mnist_slic")
class MnistSlic(Dataset):
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

        with open(config.filepath.image, "rb") as f:
            # First 16 bytes contain some metadata
            _ = f.read(4)
            size = struct.unpack(">I", f.read(4))[0]
            _ = f.read(8)
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, 28, 28)
        # Labels
        with open(config.filepath.labels, "rb") as f:
            # First 8 bytes contain some metadata
            _ = f.read(8)
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.graph_transform is not None:
            graph = self.graph_transform(image)

        if self.image_transform is not None:
            image = self.image_transform(image)
        # plt.imsave('test_img.png', np.squeeze(graph.img))
        # plt.imsave('test_seg.png', np.squeeze(graph.seg))
        # visualize_geometric_graph(graph)
        # time.sleep(5)

        return {
            "graph": graph,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }
