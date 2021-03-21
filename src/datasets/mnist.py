"""Implements MNIST Dataset"""
import struct

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from src.modules.transforms import *
from src.utils.mapper import configmapper


@configmapper.map("datasets", "mnist")
class Mnist(Dataset):
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
        if self.transform is not None:
            image = self.transform(image)

        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}
