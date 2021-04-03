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


def visualize_geometric_graph(graph, file_name="Graph.png"):
    # print("Pos",graph.pos.shape)
    # print("X", graph.x.shape)
    # print("Edge Index",graph.edge_index.shape)
    x, edge_index, pos = graph.x.numpy(), graph.edge_index.numpy(), graph.pos.numpy()

    colors = np.squeeze(x)
    src = edge_index[0]
    dst = edge_index[1]
    edgelist = list(zip(src, dst))

    # print(edgelist)
    # print(edgelist)
    g = nx.Graph()
    g.add_edges_from(edgelist)

    # print(g.nodes)
    pos_dic = dict(zip(list(range(0, colors.shape[0])), switch_coordinates(pos)))
    nx.draw_networkx(g, pos=pos_dic, node_color=colors)
    # base = plt.gca().transData
    # tr = transforms.Affine2D().rotate_deg(90)

    # For MNIST SLIC
    # pos_dic = dict(zip(list(range(0, colors.shape[0])), rotate_coordinates_by_90(switch_coordinates(pos))))

    # For RAG
    # pos_dic = dict(
    #     zip(
    #         list(range(0, colors.shape[0])),
    #         horizontal_flip(rotate_coordinates_by_90(rotate_coordinates_by_90(pos))),
    #     )
    # )

    # nx.draw_networkx(
    #     g,
    #     pos=pos_dic,
    #     node_color=colors,
    # )
    
    plt.savefig(file_name, format="PNG")
    plt.clf()
