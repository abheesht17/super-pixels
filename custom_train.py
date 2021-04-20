"""Train File."""
# Imports
import argparse
import os

from src.datasets import *
from src.models import *
from src.trainers import *
from src.utils.configuration import Config
from src.utils.logger import Logger
from src.utils.mapper import configmapper
from src.utils.misc import generate_grid_search_configs, seed


dirname = os.path.dirname(__file__)  # For Paths Relative to Current File

# Config
parser = argparse.ArgumentParser(
    prog="train.py", description="Train a model with Base Trainer."
)
parser.add_argument(
    "--config_dir", type=str, action="store", help="The directory for all config files."
)

parser.add_argument(
    "--grid_search",
    action="store_true",
    help="Whether to do a grid_search",
    default=False,
)
parser.add_argument(
    "--validation",
    action='store_true',
    help="Whether to use validation data or test data",
    default=False,
)
args = parser.parse_args()
model_config = Config(path=os.path.join(args.config_dir, "model.yaml"))
train_config = Config(path=os.path.join(args.config_dir, "train.yaml"))
data_config = Config(path=os.path.join(args.config_dir, "dataset.yaml"))
grid_search = args.grid_search
# Seed
seed(train_config.main_config.seed)

# Data
if "main" in data_config.as_dict().keys():  # Regular Data
    train_data_config = data_config.train
    val_data_config = data_config.val
    if not args.validation:
        train_filepath_config = train_data_config.filepath
        train_filepath_config.set_value('indices_csv', None)
        train_data_config.set_value('filepath',train_filepath_config) 
        
        val_filepath_config = val_data_config.filepath
        val_filepath_config.set_value('indices_csv', None)
        val_data_config.set_value('filepath',val_filepath_config) 

    train_data = configmapper.get_object("datasets", train_data_config.name)(
        train_data_config
    )
    val_data = configmapper.get_object("datasets", val_data_config.name)(
        val_data_config
    )

else:  # HF Type Data
    dataset = configmapper.get_object("datasets", data_config.name)(data_config)
    train_data = dataset.train_dataset["train"]
    val_data = dataset.train_dataset["test"]


# Logger

logger = Logger(
    log_path=os.path.join(
        "./logs/",
        args.config_dir.strip("/").split("/")[-1]
    )
)

if grid_search:
    train_configs = generate_grid_search_configs(train_config, train_config.grid_search)
    print(f"Total Configurations Generated: {len(train_configs)}")

    for train_config in train_configs:
        print(train_config)
        

        ## Seed
        seed(train_config.main_config.seed)

        model = configmapper.get_object("models", model_config.name)(model_config)
        # Trainer
        trainer = configmapper.get_object("trainers", train_config.trainer_name)(
            train_config
        )

        ## Train
        trainer.train(model, train_data, val_data, logger)

else:
    ## Seed
    seed(train_config.main_config.seed)

    model = configmapper.get_object("models", model_config.name)(model_config)


    ## Trainer
    trainer = configmapper.get_object("trainers", train_config.trainer_name)(
        train_config
    )

    ## Train
    trainer.train(model, train_data, val_data, logger)
