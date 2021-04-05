"""Train File."""
# Imports
import argparse
import os

from omegaconf import OmegaConf

from src.datasets import *
from src.models import *
from src.trainers import *
from src.utils.logger import Logger
from src.utils.mapper import configmapper
from src.utils.misc import seed


dirname = os.path.dirname(__file__)  # For Paths Relative to Current File

# Config
parser = argparse.ArgumentParser(
    prog="train.py", description="Train a model with Base Trainer."
)
parser.add_argument(
    "--config_dir", type=str, action="store", help="The directory for all config files."
)

args = parser.parse_args()
model_config = OmegaConf.load(os.path.join(args.config_dir, "model.yaml"))
train_config = OmegaConf.load(os.path.join(args.config_dir, "train.yaml"))
data_config = OmegaConf.load(os.path.join(args.config_dir, "dataset.yaml"))

# Seed
seed(train_config.main_config.seed)


# Data
if "main" in dict(data_config).keys():  # Regular Data
    train_data_config = data_config.train
    val_data_config = data_config.val
    train_data = configmapper.get_object("datasets", train_data_config.name)(
        train_data_config
    )
    val_data = configmapper.get_object("datasets", val_data_config.name)(
        val_data_config
    )

else:  # HF Type Data
    dataset = configmapper.get_object(
        "datasets", data_config.name)(data_config)
    train_data = dataset.train_dataset["train"]
    val_data = dataset.train_dataset["test"]

# Logger

logger = Logger(log_path=os.path.join(
    "/content/drive/MyDrive/SuperPixels/logs/", args.config_dir.strip('/').split('/')[-1]))

# Model
model = configmapper.get_object("models", model_config.name)(model_config)

print(model)
# Trainer
trainer = configmapper.get_object(
    "trainers", train_config.trainer_name)(train_config)

# Train
trainer.train(model, train_data, val_data, logger)
