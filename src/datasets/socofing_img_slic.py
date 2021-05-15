"""Sokoto Coventry Fingerprint Dataset (SOCOFing) Image+SLIC"""
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


@configmapper.map("datasets", "socofing_img_slic")
class SocofingImgSlic(Dataset):
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

        if config.filepath.indices_csv != None:
            data_path = config.filepath.indices_csv
        else:
            data_path = config.filepath.data
        
        self.dir_path = '/'.join(config.filepath.data.split('/')[:-1])
        self.data = pd.read_csv(data_path)
        self.image_paths = np.array(self.data["path"])
        self.labels = np.array(self.data["img_id"])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.dir_path,self.image_paths[idx]), 0)
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
