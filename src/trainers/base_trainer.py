import math
import os

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader
from tqdm import tqdm

from src.modules.losses import *
from src.modules.metrics import *
from src.modules.optimizers import *
from src.modules.schedulers import *
from src.utils.configuration import Config
from src.utils.logger import Logger
from src.utils.mapper import configmapper
from src.utils.misc import get_item_in_config


@configmapper.map("trainers", "base_trainer")
class BaseTrainer:
    def __init__(self, config):
        self._config = config
        self.metrics = {
            configmapper.get_object("metrics", metric["type"]): metric["params"]
            for metric in self._config.main_config.metrics
        }
        self.train_config = self._config.train
        self.val_config = self._config.val
        self.log_label = self.train_config.log.log_label
        self.device = torch.device(self._config.main_config.device.name)
        if self.train_config.log_and_val_interval is not None:
            self.train_config.val_interval = self.train_config.log_and_val_interval
            self.train_config.log.log_interval = self.train_config.log_and_val_interval
        print("Logging with label: ", self.log_label)

    def train(self, model, train_dataset, val_dataset=None, logger=None):
        model.to(self.device)
        optim_params = self.train_config.optimizer.params
        if optim_params:
            optimizer = configmapper.get_object(
                "optimizers", self.train_config.optimizer.type
            )(model.parameters(), **optim_params.as_dict())
        else:
            optimizer = configmapper.get_object(
                "optimizers", self.train_config.optimizer.type
            )(model.parameters())

        if self.train_config.scheduler is not None:
            scheduler_params = self.train_config.scheduler.params
            if scheduler_params:
                scheduler = configmapper.get_object(
                    "schedulers", self.train_config.scheduler.type
                )(optimizer, **scheduler_params.as_dict())
            else:
                scheduler = configmapper.get_object(
                    "schedulers", self.train_config.scheduler.type
                )(optimizer)

        criterion_params = self.train_config.criterion.params
        if criterion_params:
            if "weight" in criterion_params.as_dict():
                criterion_params.set_value("weight",torch.tensor(criterion_params.weight).to(self.device))
            criterion = configmapper.get_object(
                "losses", self.train_config.criterion.type
            )(**criterion_params.as_dict())
        else:
            criterion = configmapper.get_object(
                "losses", self.train_config.criterion.type
            )()
        if self._config.dataloader_type == "geometric":
            train_loader = GeometricDataLoader(
                train_dataset, **self.train_config.loader_params.as_dict()
            )
        else:
            train_loader = DataLoader(
                dataset=train_dataset, **self.train_config.loader_params.as_dict()
            )

        max_epochs = self.train_config.max_epochs
        batch_size = self.train_config.loader_params.batch_size
        interval_type = self.train_config.interval_type
        val_interval = self.train_config.val_interval
        log_interval = self.train_config.log.log_interval

        if logger is None:
            train_logger = Logger(**self.train_config.log.logger_params.as_dict())
        else:
            train_logger = logger

        train_log_values = self.train_config.log.vals.as_dict()

        best_score = (
            -math.inf if self.train_config.save_on.desired == "max" else math.inf
        )
        save_on_score = self.train_config.save_on.score
        best_step = -1

        best_hparam_list = None
        best_hparam_name_list = None
        best_metrics_list = None
        best_metrics_name_list = None

        # print("\nTraining\n")
        # print(max_steps)

        global_step = 0
        for epoch in range(1, max_epochs + 1):
            print(
                "Epoch: {}/{}, Global Step: {}".format(epoch, max_epochs, global_step)
            )
            train_loss = 0
            if self.train_config.label_type == "float":
                all_labels = torch.FloatTensor().to(self.device)
            else:
                all_labels = torch.LongTensor().to(self.device)

            all_outputs = torch.Tensor().to(self.device)

            train_scores = None
            val_scores = None

            pbar = tqdm(total=math.ceil(len(train_dataset) / batch_size))
            pbar.set_description("Epoch " + str(epoch))

            for step, batch in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                for key in batch:
                    batch[key] = batch[key].to(self.device)

                inputs = {}
                for key in self._config.input_key:
                    inputs[key] = batch[key]
                labels = batch["label"]

                # NOW THIS MUST BE HANDLED IN THE DATASET CLASS
                # if self.train_config.label_type == "float":
                # # Specific to Float Type
                #     labels = labels.float()

                outputs = model(**inputs)

                # Can remove this at a later stage?
                # I think `losses.backward()` should work.
                loss = criterion(torch.squeeze(outputs, dim=1), labels)
                loss.backward()

                all_labels = torch.cat((all_labels, labels), 0)

                if self.train_config.label_type == "float":
                    all_outputs = torch.cat((all_outputs, outputs), 0)
                else:
                    all_outputs = torch.cat(
                        (all_outputs, torch.argmax(outputs, axis=1)), 0
                    )

                train_loss += loss.item()
                optimizer.step()

                if self.train_config.scheduler is not None:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(train_loss / (step + 1))
                    else:
                        scheduler.step()

                # print(train_loss)
                # print(step+1)

                pbar.set_postfix_str(f"Train Loss: {train_loss /(step+1)}")
                pbar.update(1)

                global_step += 1

                # Need to check if we want global_step or local_step
                if interval_type == "step":
                    if (
                        val_dataset is not None
                        and (global_step - 1) % val_interval == 0
                    ):
                        # print("\nEvaluating\n")
                        val_scores = self.val(
                            model,
                            val_dataset,
                            global_step,
                            train_logger,
                            train_log_values,
                        )

                        # save_flag = 0
                        if self.train_config.save_on is not None:

                            # BEST SCORES UPDATING

                            train_scores = self.get_scores(
                                train_loss,
                                global_step,
                                self.train_config.criterion.type,
                                all_outputs,
                                all_labels,
                            )

                            best_score, best_step, save_flag = self.check_best(
                                val_scores, save_on_score, best_score, global_step
                            )

                            store_dict = {
                                "model_state_dict": model.state_dict(),
                                "best_step": best_step,
                                "best_score": best_score,
                                "save_on_score": save_on_score,
                            }

                            path = os.path.join(
                                train_logger.log_path,
                                self.train_config.save_on.best_path.format(
                                    self.log_label
                                ),
                            )

                            self.save(store_dict, path, save_flag)

                            if save_flag and train_log_values["hparams"] is not None:
                                (
                                    best_hparam_list,
                                    best_hparam_name_list,
                                    best_metrics_list,
                                    best_metrics_name_list,
                                ) = self.update_hparams(
                                    train_scores, val_scores, desc="best_val"
                                )
                    # pbar.close()
                    if (global_step - 1) % log_interval == 0:
                        # print("\nLogging\n")
                        train_loss_name = self.train_config.criterion.type
                        metric_list = [
                            metric(
                                all_labels.cpu(),
                                all_outputs.detach().cpu(),
                                **self.metrics[metric],
                            )
                            for metric in self.metrics
                        ]
                        metric_name_list = [
                            metric["type"]
                            for metric in self._config.main_config.metrics
                        ]

                        train_scores = self.log(
                            train_loss / (step + 1),
                            train_loss_name,
                            metric_list,
                            metric_name_list,
                            train_logger,
                            train_log_values,
                            global_step,
                            append_text=self.train_config.append_text,
                        )
            pbar.close()
            if not os.path.exists(
                os.path.join(
                    train_logger.log_path, self.train_config.checkpoint.checkpoint_dir
                )
            ):
                os.makedirs(
                    os.path.join(
                        train_logger.log_path,
                        self.train_config.checkpoint.checkpoint_dir,
                    )
                )

            if self.train_config.save_after_epoch:
                store_dict = {
                    "model_state_dict": model.state_dict(),
                }

                path = f"{os.path.join(train_logger.log_path, self.train_config.checkpoint.checkpoint_dir)}epoch_{str(self.train_config.log.log_label)}_{str(epoch)}.pth"

                self.save(store_dict, path, save_flag=1)
            if interval_type == "epoch":
                if val_dataset is not None and (epoch) % val_interval == 0:
                    # print("\nEvaluating\n")
                    val_scores = self.val(
                        model,
                        val_dataset,
                        epoch,
                        train_logger,
                        train_log_values,
                    )

                    # save_flag = 0
                    if self.train_config.save_on is not None:

                        # BEST SCORES UPDATING

                        train_scores = self.get_scores(
                            train_loss,
                            epoch,
                            self.train_config.criterion.type,
                            all_outputs,
                            all_labels,
                        )

                        best_score, best_epoch, save_flag = self.check_best(
                            val_scores, save_on_score, best_score, epoch
                        )

                        store_dict = {
                            "model_state_dict": model.state_dict(),
                            "best_epoch": best_epoch,
                            "best_score": best_score,
                            "save_on_score": save_on_score,
                        }

                        path = os.path.join(
                            train_logger.log_path,
                            self.train_config.save_on.best_path.format(self.log_label),
                        )

                        self.save(store_dict, path, save_flag)

                        if save_flag and train_log_values["hparams"] is not None:
                            (
                                best_hparam_list,
                                best_hparam_name_list,
                                best_metrics_list,
                                best_metrics_name_list,
                            ) = self.update_hparams(
                                train_scores, val_scores, desc="best_val"
                            )

                # pbar.close()
                if (epoch) % log_interval == 0:
                    # print("\nLogging\n")
                    train_loss_name = self.train_config.criterion.type
                    metric_list = [
                        metric(
                            all_labels.cpu(),
                            all_outputs.detach().cpu(),
                            **self.metrics[metric],
                        )
                        for metric in self.metrics
                    ]
                    metric_name_list = [
                        metric["type"] for metric in self._config.main_config.metrics
                    ]

                    train_scores = self.log(
                        train_loss / len(train_loader),
                        train_loss_name,
                        metric_list,
                        metric_name_list,
                        train_logger,
                        train_log_values,
                        epoch,
                        append_text=self.train_config.append_text,
                    )
            if epoch == max_epochs:
                # print("\nEvaluating\n")
                if interval_type == "step":
                    val_scores = self.val(
                        model,
                        val_dataset,
                        global_step,
                        train_logger,
                        train_log_values,
                    )

                    # print("\nLogging\n")
                    train_loss_name = self.train_config.criterion.type
                    metric_list = [
                        metric(
                            all_labels.cpu(),
                            all_outputs.detach().cpu(),
                            **self.metrics[metric],
                        )
                        for metric in self.metrics
                    ]
                    metric_name_list = [
                        metric["type"] for metric in self._config.main_config.metrics
                    ]

                    train_scores = self.log(
                        train_loss / len(train_loader),
                        train_loss_name,
                        metric_list,
                        metric_name_list,
                        train_logger,
                        train_log_values,
                        global_step,
                        append_text=self.train_config.append_text,
                    )

                if self.train_config.save_on is not None:

                    # BEST SCORES UPDATING

                    train_scores = self.get_scores(
                        train_loss,
                        len(train_loader),
                        self.train_config.criterion.type,
                        all_outputs,
                        all_labels,
                    )

                    best_score, best_step, save_flag = self.check_best(
                        val_scores, save_on_score, best_score, global_step
                    )

                    store_dict = {
                        "model_state_dict": model.state_dict(),
                        "best_step": best_step,
                        "best_score": best_score,
                        "save_on_score": save_on_score,
                    }

                    path = os.path.join(
                        train_logger.log_path,
                        self.train_config.save_on.best_path.format(self.log_label),
                    )

                    self.save(store_dict, path, save_flag)

                    if save_flag and train_log_values["hparams"] is not None:
                        (
                            best_hparam_list,
                            best_hparam_name_list,
                            best_metrics_list,
                            best_metrics_name_list,
                        ) = self.update_hparams(
                            train_scores, val_scores, desc="best_val"
                        )

                    # FINAL SCORES UPDATING + STORING
                    train_scores = self.get_scores(
                        train_loss,
                        len(train_loader),
                        self.train_config.criterion.type,
                        all_outputs,
                        all_labels,
                    )

                    store_dict = {
                        "model_state_dict": model.state_dict(),
                        "final_step": global_step,
                        "final_score": train_scores[save_on_score],
                        "save_on_score": save_on_score,
                    }

                    path = os.path.join(
                        train_logger.log_path,
                        self.train_config.save_on.final_path.format(self.log_label),
                    )

                    self.save(store_dict, path, save_flag=1)
                    if train_log_values["hparams"] is not None:
                        (
                            final_hparam_list,
                            final_hparam_name_list,
                            final_metrics_list,
                            final_metrics_name_list,
                        ) = self.update_hparams(train_scores, val_scores, desc="final")
                        train_logger.save_hyperparams(
                            best_hparam_list,
                            best_hparam_name_list,
                            [
                                int(self.log_label),
                            ]
                            + best_metrics_list
                            + final_metrics_list,
                            [
                                "hparams/log_label",
                            ]
                            + best_metrics_name_list
                            + final_metrics_name_list,
                        )
                    #

    # Need to check if we want same loggers of different loggers for train and eval
    # Evaluate

    def get_scores(self, loss, divisor, loss_name, all_outputs, all_labels):

        avg_loss = loss / divisor

        metric_list = [
            metric(all_labels.cpu(), all_outputs.detach().cpu(), **self.metrics[metric])
            for metric in self.metrics
        ]
        metric_name_list = [
            metric["type"] for metric in self._config.main_config.metrics
        ]

        return dict(zip([loss_name] + metric_name_list, [avg_loss] + metric_list))

    def check_best(self, val_scores, save_on_score, best_score, global_step):
        save_flag = 0
        best_step = global_step
        if self.train_config.save_on.desired == "min":
            if val_scores[save_on_score] < best_score:
                save_flag = 1
                best_score = val_scores[save_on_score]
                best_step = global_step
        else:
            if val_scores[save_on_score] > best_score:
                save_flag = 1
                best_score = val_scores[save_on_score]
                best_step = global_step
        return best_score, best_step, save_flag

    def update_hparams(self, train_scores, val_scores, desc):
        hparam_list = []
        hparam_name_list = []
        for hparam in self.train_config.log.vals.hparams:
            hparam_list.append(get_item_in_config(self._config, hparam["path"]))
            if isinstance(hparam_list[-1], Config):
                hparam_list[-1] = hparam_list[-1].as_dict()
            hparam_name_list.append(hparam["name"])

        val_keys, val_values = zip(*val_scores.items())
        train_keys, train_values = zip(*train_scores.items())
        val_keys = list(val_keys)
        train_keys = list(train_keys)
        val_values = list(val_values)
        train_values = list(train_values)
        for i, key in enumerate(val_keys):
            val_keys[i] = f"hparams/{desc}_val_" + val_keys[i]
        for i, key in enumerate(train_keys):
            train_keys[i] = f"hparams/{desc}_train_" + train_keys[i]
        # train_logger.save_hyperparams(hparam_list, hparam_name_list,\
        # train_values+val_values,train_keys+val_keys, )
        return (
            hparam_list,
            hparam_name_list,
            train_values + val_values,
            train_keys + val_keys,
        )

    def save(self, store_dict, path, save_flag=0):
        if save_flag:
            dirs = "/".join(path.split("/")[:-1])
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            torch.save(store_dict, path)

    def log(
        self,
        loss,
        loss_name,
        metric_list,
        metric_name_list,
        logger,
        log_values,
        global_step,
        append_text,
    ):

        return_dic = dict(
            zip(
                [
                    loss_name,
                ]
                + metric_name_list,
                [
                    loss,
                ]
                + metric_list,
            )
        )

        loss_name = f"{append_text}_{self.log_label}_{loss_name}"
        if log_values["loss"]:
            logger.save_params(
                [loss],
                [loss_name],
                combine=True,
                combine_name="losses",
                global_step=global_step,
            )

        for i in range(len(metric_name_list)):
            metric_name_list[
                i
            ] = f"{append_text}_{self.log_label}_{metric_name_list[i]}"
        if log_values["metrics"]:
            logger.save_params(
                metric_list,
                metric_name_list,
                combine=True,
                combine_name="metrics",
                global_step=global_step,
            )
            # print(hparams_list)
            # print(hparam_name_list)

        # for k,v in dict(zip([loss_name],[loss])).items():
        #     print(f"{k}:{v}")
        # for k,v in dict(zip(metric_name_list,metric_list)).items():
        #     print(f"{k}:{v}")
        return return_dic

    def val(
        self,
        model,
        dataset,
        global_step,
        train_logger=None,
        train_log_values=None,
        log=True,
    ):
        append_text = self.val_config.append_text
        criterion_params = self.train_config.criterion.params
        if criterion_params:
            if "weight" in criterion_params.as_dict():
                criterion_params.set_value("weight",torch.tensor(criterion_params.weight).to(self.device))
            criterion = configmapper.get_object(
                "losses", self.train_config.criterion.type
            )(**criterion_params.as_dict())
        else:
            criterion = configmapper.get_object(
                "losses", self.train_config.criterion.type
            )()
        if train_logger is not None:
            val_logger = train_logger
        else:
            val_logger = Logger(**self.val_config.log.logger_params.as_dict())

        if train_log_values is not None:
            val_log_values = train_log_values
        else:
            val_log_values = self.val_config.log.vals.as_dict()

        if self._config.dataloader_type == "geometric":
            val_loader = GeometricDataLoader(
                dataset, **self.val_config.loader_params.as_dict()
            )
        else:
            val_loader = DataLoader(
                dataset=dataset, **self.val_config.loader_params.as_dict()
            )

        all_outputs = torch.Tensor().to(self.device)
        if self.train_config.label_type == "float":
            all_labels = torch.FloatTensor().to(self.device)
        else:
            all_labels = torch.LongTensor().to(self.device)

        with torch.no_grad():
            model.eval()
            val_loss = 0
            for j, batch in enumerate(val_loader):
                for key in batch:
                    batch[key] = batch[key].to(self.device)

                inputs = {}
                for key in self._config.input_key:
                    inputs[key] = batch[key]
                labels = batch["label"]

                # NOW THIS MUST BE HANDLED IN THE DATASET CLASS
                # if self.train_config.label_type == "float":
                # # Specific to Float Type
                #     labels = labels.float()

                outputs = model(**inputs)

                loss = criterion(torch.squeeze(outputs, dim=1), labels)
                val_loss += loss.item()

                all_labels = torch.cat((all_labels, labels), 0)

                if self.train_config.label_type == "float":
                    all_outputs = torch.cat((all_outputs, outputs), 0)
                else:
                    all_outputs = torch.cat(
                        (all_outputs, torch.argmax(outputs, axis=1)), 0
                    )

            val_loss = val_loss / len(val_loader)

            val_loss_name = self.train_config.criterion.type

            # print(all_outputs, all_labels)
            metric_list = [
                metric(
                    all_labels.cpu(), all_outputs.detach().cpu(), **self.metrics[metric]
                )
                for metric in self.metrics
            ]
            metric_name_list = [
                metric["type"] for metric in self._config.main_config.metrics
            ]
            return_dic = dict(
                zip(
                    [
                        val_loss_name,
                    ]
                    + metric_name_list,
                    [
                        val_loss,
                    ]
                    + metric_list,
                )
            )
            if log:
                val_scores = self.log(
                    val_loss,
                    val_loss_name,
                    metric_list,
                    metric_name_list,
                    val_logger,
                    val_log_values,
                    global_step,
                    append_text,
                )
                return val_scores
            return return_dic
