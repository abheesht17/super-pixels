"""Implements COVID Img Slic sDataset"""

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.modules.transforms import *
from src.utils.mapper import configmapper


@configmapper.map("datasets", "covid_img_slic")
class CovidImgSlic(Dataset):
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

        self.data_paths_df = pd.read_csv(config.data_paths_csv)
        self.data_paths_df["path"] = self.data_paths_df["path"].apply(
            lambda x: os.path.join("/".join(config.data_paths_csv.split("/")[:-1]), x)
        )

    def __len__(self):
        return self.data_paths_df.shape[0]

    def __getitem__(self, idx):
        image = Image.open(self.data_paths_df.iloc[idx]["path"]).convert("L")
        label = self.data_paths_df.iloc[idx]["label"]
        if self.graph_transform is not None:
            graph = self.graph_transform(image)

        if self.image_transform is not None:
            image = self.image_transform(image)

        return {
            "graph": graph,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }
