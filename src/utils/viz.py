import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import transforms
from scipy.ndimage.interpolation import rotate


def switch_coordinates(coordinates_list):
    a = coordinates_list[:, 0]
    b = coordinates_list[:, 1]
    return np.array(list(zip(b, a)))


def rotate_coordinates_by_90(coordinates_list):
    a = coordinates_list[:, 0]
    a = np.max(a) - a
    a = (a - np.min(a)) / (np.max(a) - np.min(a))
    b = coordinates_list[:, 1]
    return np.array(list(zip(b, a)))


def horizontal_flip(coordinates_list):
    a = coordinates_list[:, 0]
    a = np.max(a) - a
    a = (a - np.min(a)) / (np.max(a) - np.min(a))
    b = coordinates_list[:, 1]
    return np.array(list(zip(a, b)))


def visualize_geometric_graph(
    graph,
    file_name="Graph.png",
    normalize_mean=None,
    normalize_std=None,
    img=None,
):
    # print("Pos",graph.pos.shape)
    # print("X", graph.x.shape)
    # print("Edge Index",graph.edge_index.shape)
    x, edge_index, pos = graph.x.numpy(), graph.edge_index.numpy(), graph.pos.numpy()

    colors = np.squeeze(x)
    if normalize_mean is not None:
        colors = np.clip(colors * normalize_std + normalize_mean, 0, 1)
    src = edge_index[0]
    dst = edge_index[1]

    edgelist = list(zip(src, dst))

    if img is not None:
        plt.imshow(img, alpha=0.7)

    g = nx.Graph()
    g.add_nodes_from(list(range(0, colors.shape[0])))
    g.add_edges_from(edgelist)

    # if len(np.unique(edgelist)) < colors.shape[0]:
    # print("="*30)
    # print(f"Edgelist has only {len(np.unique(edgelist))} unique nodes.")
    # print(f"However, total number of nodes is {colors.shape[0]}.")

    # existing_nodes = np.sort(np.unique(edgelist))

    pos_dic = dict(zip(list(range(0, colors.shape[0])), pos))
    nx.draw_networkx(
        g,
        pos=pos_dic,
        node_color=colors,
        node_size=150,
        font_size=7,
        edgecolors="black",
    )
    # nx.draw_networkx(g, pos=pos_dic, node_color=colors[existing_nodes], labels = dict(zip(existing_nodes, existing_nodes)))

    # for element in list(range(0,colors.shape[0])):
    #     if element not in existing_nodes:
    #         # print(f"Check for node {element} in {file_name}.")
    #         plt.scatter(x=[pos_dic[element][1]], y=[pos_dic[element][0]], c = colors[element], label = str(element), s = 300)
    #         plt.text(x=pos_dic[element][1]-0.5, y=pos_dic[element][0]-0.5, s=str(element), fontsize=12)

    plt.savefig(file_name, bbox_inches="tight")
    plt.clf()
