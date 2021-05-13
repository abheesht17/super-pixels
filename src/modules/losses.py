"All criterion functions."
import torch
from torch.nn import CrossEntropyLoss, MSELoss

from src.utils.mapper import configmapper


configmapper.map("losses", "mse")(MSELoss)
configmapper.map("losses", "CrossEntropyLoss")(CrossEntropyLoss)


@configmapper.map("losses", "HybridLoss")
class HybridLoss:
    def __init__(self, alpha=0.75, weight=None):
        if weight is None:
        	self.loss_fn = CrossEntropyLoss()
        else:
            weight = torch.tensor(weight)
        	self.loss_fn  = CrossEntropyLoss(weight=weight)
        self.alpha = alpha

    def __call__(self, outputs, targets):
        return self.loss_fn(
            torch.squeeze(outputs[0], dim=1), targets
        ) * self.alpha + self.loss_fn(torch.squeeze(outputs[1], dim=1), targets) * (
            1 - self.alpha
        )
