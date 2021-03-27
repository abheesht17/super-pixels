import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.transforms as torch_geometric_transforms
import torchvision.transforms as transforms
from skimage.segmentation import slic
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

# https://github.com/phcavelar/mnist-superpixel
@configmapper.map("transforms", "RAGGraph")
class RAGGraph(object):
    def __init__(self, add_seg=False, add_img=False, **kwargs):
        self.add_seg = add_seg
        self.add_img = add_img
        self.kwargs = kwargs

    def __call__(self, img):
        img = img.permute(1, 2, 0)
        h, w, c = img.size()
        seg = slic(
            img.to(torch.double).numpy(),
            slic_zero=True,
            **self.kwargs,  # provide n_segments
        )
        # print(seg)

        aseg = np.array(seg)
        num_nodes = np.max(aseg)

        nodes = {
            node: {"rgb_list": [], "pos_list": []} for node in range(num_nodes + 1)
        }
        img = img.to(torch.double).numpy()

        height = img.shape[0]
        width = img.shape[1]
        for y in range(height):
            for x in range(width):
                node = aseg[y, x]
                rgb = img[y, x, :]
                pos = np.array([float(x) / width, float(y) / height])
                nodes[node]["rgb_list"].append(rgb)
                nodes[node]["pos_list"].append(pos)

        x = []
        pos = []
        for node in nodes:
            nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
            nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
            # rgb
            rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
            # rgb_std = np.std(nodes[node]["rgb_list"], axis=0)
            # rgb_gram = np.matmul( nodes[node]["rgb_list"].T, nodes[node]["rgb_list"] ) / nodes[node]["rgb_list"].shape[0]
            # Pos
            pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
            # pos_std = np.std(nodes[node]["pos_list"], axis=0)
            # pos_gram = np.matmul( nodes[node]["pos_list"].T, nodes[node]["pos_list"] ) / nodes[node]["pos_list"].shape[0]
            # Debug

            features = np.concatenate(
                [
                    np.reshape(rgb_mean, -1),
                    # np.reshape(rgb_std, -1),
                    # np.reshape(rgb_gram, -1),
                    # np.reshape(pos_mean, -1),
                    # np.reshape(pos_std, -1),
                    # np.reshape(pos_gram, -1)
                ]
            )
            x.append(features)
            pos.append(pos_mean)

            # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments

        # centers
        vs_right = np.vstack([seg[:, :-1].ravel(), seg[:, 1:].ravel()])
        vs_below = np.vstack([seg[:-1, :].ravel(), seg[1:, :].ravel()])
        bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

        from_indices = []
        to_indices = []

        # Adjacency loops
        for i in range(bneighbors.shape[1]):
            if bneighbors[0, i] != bneighbors[1, i]:
                from_indices += [bneighbors[0, i], bneighbors[1, i]]
                to_indices += [bneighbors[1, i], bneighbors[0, i]]

        # Self loops
        for node in nodes:
            from_indices += [node]
            to_indices += [node]

        data = Data(
            x=torch.tensor(x, dtype=torch.float),
            pos=torch.tensor(pos, dtype=torch.float),
            edge_index=torch.tensor([from_indices, to_indices], dtype=torch.long),
        )

        if self.add_seg:
            seg = torch.from_numpy(seg)
            data.seg = seg.view(1, h, w)

        if self.add_img:
            img = torch.from_numpy(img)
            data.img = img.permute(2, 0, 1).view(1, c, h, w)

        return data


@configmapper.map("transforms", "MnistSLIC")
class MnistSLIC(object):
    def __init__(self, add_seg=False, add_img=False, **kwargs):
        self.add_seg = add_seg
        self.add_img = add_img
        self.kwargs = kwargs

    def __call__(self, img):
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
