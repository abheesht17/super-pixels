"""Implements COVID SLIC Dataset"""

import os

import pandas as pd
import torch
from PIL import Image

# import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from src.modules.transforms import *
from src.utils.mapper import configmapper


# from src.utils.viz import visualize_geometric_graph
# import matplotlib.pyplot as plt


@configmapper.map("datasets", "covid_slic")
class CovidSlic(Dataset):
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

        self.data_paths_df = pd.read_csv(config.data_paths_csv)
        self.data_paths_df["path"] = self.data_paths_df["path"].apply(
            lambda x: os.path.join("/".join(config.data_paths_csv.split("/")[:-1]), x)
        )

    def __len__(self):
        return self.data_paths_df.shape[0]

    def __getitem__(self, idx):
        image = Image.open(self.data_paths_df.iloc[idx]["path"]).convert("L")
        label = self.data_paths_df.iloc[idx]["label"]
        if self.transform is not None:
            graph = self.transform(image)

        # pix = np.array(image.resize((256,256)).getdata()).reshape(256,256)
        # plt.imsave(f"covid_{label}.pdf", pix)
        # visualize_geometric_graph(graph, f"covid_graph_{label}.pdf",0.5094362248497217, 0.25237422057401465, pix)

        return {"graph": graph, "label": torch.tensor(label, dtype=torch.long)}
