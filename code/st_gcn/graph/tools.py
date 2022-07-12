from typing import List

import numpy as np


def edge2mat(link: List[tuple], num_node: int) -> np.ndarray:
    """
    Creates a sparse matrix with all the linkages

    Args:
        link (List[tuple]): landmark connections
        num_node (int): Total number of landmarks

    Returns:
        np.ndarray: sparse matrix containing mentioning all the on-linkages
    """
    connec_matrix = np.zeros((num_node, num_node))
    for i, j in link:
        connec_matrix[j, i] = 1
    return connec_matrix


def normalize_digraph(A: np.ndarray) -> np.ndarray:
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return np.dot(A, Dn)


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    return np.dot(np.dot(Dn, A), Dn)


def get_uniform_graph(num_node: int, self_link: List[tuple], neighbor):
    return normalize_digraph(edge2mat(neighbor + self_link, num_node))


def get_uniform_distance_graph(num_node: int, neighbor):
    I = np.eye(num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    return I - N


def get_distance_graph(num_node: int, neighbor):
    I = np.eye(num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    return np.stack((I, N))


def get_spatial_graph(
    num_node: int, inward: List[tuple], outward: List[tuple]
) -> np.ndarray:
    I = np.eye(num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    return np.stack((I, In, Out))


def get_DAD_graph(num_node: int, self_link: List[tuple], neighbor):
    return normalize_undigraph(edge2mat(neighbor + self_link, num_node))


def get_DLD_graph(num_node: int, neighbor):
    I = np.eye(num_node)
    return I - normalize_undigraph(edge2mat(neighbor, num_node))
