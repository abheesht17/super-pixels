"All criterion functions."
from torch.nn import CrossEntropyLoss, MSELoss

from src.utils.mapper import configmapper
import torch

configmapper.map("losses", "mse")(MSELoss)
configmapper.map("losses", "CrossEntropyLoss")(CrossEntropyLoss)


@configmapper.map("losses", "HybridLoss")
class HybridLoss:
    def __init__(self, alpha=2):
        self.loss_fn = CrossEntropyLoss()
        self.alpha = alpha

    def __call__(self, outputs, targets):
        return (
            self.loss_fn(torch.squeeze(outputs[0], dim=1), targets)
            + self.loss_fn(torch.squeeze(outputs[1], dim=1), targets) * self.alpha
        )
