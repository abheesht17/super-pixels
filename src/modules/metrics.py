"""Metrics."""
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from datasets import load_metric
import numpy as np

from src.utils.mapper import configmapper

configmapper.map("metrics", "sklearn_f1")(f1_score)
configmapper.map("metrics", "sklearn_p")(precision_score)
configmapper.map("metrics", "sklearn_r")(recall_score)
configmapper.map("metrics", "sklearn_roc")(roc_auc_score)
configmapper.map("metrics", "sklearn_acc")(accuracy_score)
configmapper.map("metrics", "sklearn_mse")(mean_squared_error)


@configmapper.map("metrics", "hf_accuracy")
class HFAccuracy:
    acc_metric = load_metric("accuracy")

    @classmethod
    def compute_metrics(cls, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return cls.acc_metric.compute(predictions=predictions, references=labels)