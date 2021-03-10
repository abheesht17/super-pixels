from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, CyclicLR,
                                      ReduceLROnPlateau, StepLR)

from src.utils.mapper import configmapper

configmapper.map("schedulers", "step")(StepLR)
configmapper.map("schedulers", "cosineanneal")(CosineAnnealingLR)
configmapper.map("schedulers", "reduceplateau")(ReduceLROnPlateau)
configmapper.map("schedulers", "cyclic")(CyclicLR)
configmapper.map("schedulers", "cosineannealrestart")(CosineAnnealingWarmRestarts)
