import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.transforms as torch_geometric_transforms
import torchvision.transforms as transforms
from torch_geometric.data import Data
from torch_scatter import scatter_mean

from src.utils.mapper import configmapper


configmapper.map("transforms", "Resize")(transforms.Resize)
configmapper.map("transforms", "Normalize")(transforms.Normalize)
configmapper.map("transforms", "ToTensor")(transforms.ToTensor)
configmapper.map("transforms", "ToPILImage")(transforms.ToPILImage)
configmapper.map("transforms", "Grayscale")(transforms.Grayscale)
configmapper.map("transforms", "ToSLIC")(
    torch_geometric_transforms.to_superpixels.ToSLIC
)
configmapper.map("transforms", "KNNGraph")(torch_geometric_transforms.KNNGraph)


@configmapper.map("transforms", "MnistSLIC")
class MnistSLIC(object):
    def __init__(self, add_seg=False, add_img=False, **kwargs):
        self.add_seg = add_seg
        self.add_img = add_img
        self.kwargs = kwargs

    def __call__(self, img):
        from skimage.segmentation import slic

        img = img.permute(1, 2, 0)
        h, w, c = img.size()
        seg = slic(
            img.to(torch.double).numpy(),
            start_label=1,
            mask=np.squeeze(img > 0.5),
            **self.kwargs,
        )
        seg = torch.from_numpy(seg)

        x = scatter_mean(img.view(h * w, c), seg.view(h * w), dim=0)

        pos_y = torch.arange(h, dtype=torch.float)
        pos_y = pos_y.view(-1, 1).repeat(1, w).view(h * w)
        pos_x = torch.arange(w, dtype=torch.float)
        pos_x = pos_x.view(1, -1).repeat(h, 1).view(h * w)

        pos = torch.stack([pos_x, pos_y], dim=-1)
        pos = scatter_mean(pos, seg.view(h * w), dim=0)

        data = Data(x=x, pos=pos)

        if self.add_seg:
            data.seg = seg.view(1, h, w)

        if self.add_img:
            data.img = img.permute(2, 0, 1).view(1, c, h, w)

        return data
