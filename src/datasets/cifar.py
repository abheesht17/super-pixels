"""Implements CIFAR Dataset"""
import struct

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.modules.transforms import *
from src.utils.mapper import configmapper
import pandas as pd
import pickle
@configmapper.map("datasets", "cifar")
class Cifar(Dataset):
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
        with open(config.filepath.data,'rb') as f:
            self.data = pickle.load(f)
        self.images = self.data[b'data']
        self.labels = self.data[b'labels']
            
        if config.filepath.indices_csv != None:
            filtered_indices = list(pd.read_csv(config.filepath.indices_csv)['index'])
            self.images = np.take(self.images, filtered_indices, axis=0)
            self.labels = np.take(self.labels, filtered_indices, axis=0)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx].reshape(32,32,3)
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)

        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}
