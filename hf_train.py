"""Train File."""
# Imports
import argparse
import os

from transformers import Trainer, TrainingArguments

from omegaconf import OmegaConf
from src.datasets import *
from src.models import *
from src.modules.metrics import *
from src.utils.mapper import configmapper
from src.utils.misc import seed

# Config

parser = argparse.ArgumentParser(
    prog="train.py", description="Train a model with HF Trainer."
)
parser.add_argument(
    "--config_dir", type=str, action="store", help="The directory for all config files."
)
# parser.add_argument(
#     "--model",
#     type=str,
#     action="store",
#     help="The configuration for model",
# )
# parser.add_argument(
#     "--train",
#     type=str,
#     action="store",
#     help="The configuration for model training/evaluation",
# )
# parser.add_argument(
#     "--data",
#     type=str,
#     action="store",
#     help="The configuration for data",
# )

args = parser.parse_args()
model_config = OmegaConf.load(os.path.join(args.config_dir, "model.yaml"))
train_config = OmegaConf.load(os.path.join(args.config_dir, "train.yaml"))
data_config = OmegaConf.load(os.path.join(args.config_dir, "dataset.yaml"))

# Seed
seed(train_config.args.seed)  # just in case

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
    dataset = configmapper.get_object("datasets", data_config.name)(data_config)
    train_data = dataset.train_dataset["train"]
    val_data = dataset.train_dataset["test"]


# Model
model = configmapper.get_object("models", model_config.name)(model_config)

args = TrainingArguments(**OmegaConf.to_container(train_config.args, resolve=True))
# Checking for Checkpoints
if not os.path.exists(train_config.args.output_dir):
    os.makedirs(train_config.args.output_dir)
checkpoints = sorted(
    os.listdir(train_config.args.output_dir), key=lambda x: int(x.split("-")[1])
)
if len(checkpoints) != 0:
    print("Found Checkpoints:")
    print(checkpoints)
# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=configmapper.get_object(
        "metrics", train_config.metric
    ).compute_metrics,
)
# Train
if len(checkpoints) != 0:
    trainer.train(
        os.path.join(train_config.args.output_dir, checkpoints[-1])
    )  # Load from checkpoint
else:
    trainer.train()
if not os.path.exists(train_config.save_model_path):
    os.makedirs(train_config.save_model_path)
trainer.save_model(train_config.save_model_path)
