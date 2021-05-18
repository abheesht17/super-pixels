"""Sokoto Coventry Fingerprint Dataset (SOCOFing)"""
import pickle
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from src.modules.transforms import *
from src.utils.mapper import configmapper


@configmapper.map("datasets", "socofing")
class Socofing(Dataset):
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
        if config.filepath.indices_csv != None:
            data_path = config.filepath.indices_csv
        else:
            data_path = config.filepath.data
        self.dir_path = config.filepath.data
        self.data = pd.read_csv(data_path)
        self.image_paths = np.array(self.data["path"])
        self.labels = np.array(self.data["img_id"])

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.dir_path,self.image_paths[idx]), 0)
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)

        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}
