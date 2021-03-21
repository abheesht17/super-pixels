"All criterion functions."
from torch.nn import CrossEntropyLoss, MSELoss

from src.utils.mapper import configmapper


configmapper.map("losses", "mse")(MSELoss)
configmapper.map("losses", "CrossEntropyLoss")(CrossEntropyLoss)
