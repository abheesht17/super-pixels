"""Train File."""
## Imports
import argparse

# import itertools
import copy
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn

from src.utils.misc import seed, generate_grid_search_configs

from src.datasets import *
from src.models import *
from src.trainers import *

from src.utils.mapper import configmapper
from src.utils.logger import Logger

import os

dirname = os.path.dirname(__file__)  ## For Paths Relative to Current File

## Config
parser = argparse.ArgumentParser(prog="train.py", description="Train a model.")
parser.add_argument(
    "--model",
    type=str,
    action="store",
    help="The configuration for model",
)
parser.add_argument(
    "--train",
    type=str,
    action="store",
    help="The configuration for model training/evaluation",
)
parser.add_argument(
    "--data",
    type=str,
    action="store",
    help="The configuration for data",
)

args = parser.parse_args()
# print(vars(args))
model_config = OmegaConf.load(path=args.model)
train_config = OmegaConf.load(path=args.train)
data_config = OmegaConf.load(path=args.data)

## Seed
seed(train_config.main_config.seed)


## Trainer
trainer = configmapper.get_object("trainers", train_config.trainer_name)(train_config)

## Train
trainer.train(model, train_data, val_data)
