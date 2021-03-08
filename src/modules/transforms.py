import torchvision.transforms as transforms
import torch_geometric.transforms as torch_geometric_transforms
from src.utils.mapper import configmapper

configmapper.map("transforms", "Resize")(transforms.Resize)
configmapper.map("transforms", "Normalize")(transforms.Normalize)
configmapper.map("transforms", "ToTensor")(transforms.ToTensor)
configmapper.map("transforms", "ToPILImage")(transforms.ToPILImage)
configmapper.map("transforms", "Grayscale")(transforms.Grayscale)
configmapper.map("transforms", "ToSLIC")(torch_geometric_transforms.to_superpixels.ToSLIC)
