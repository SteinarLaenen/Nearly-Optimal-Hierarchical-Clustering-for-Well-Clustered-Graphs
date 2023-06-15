"""
This module is designed to generate several networkx graphs corresponding to synthetic and real-world data
"""
import networkx as nx
import random
import math

from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
#############################################################
# Synthetic graph generators
#############################################################


def generate_Gnp_graph(n, p):
    """
    Given a number of vertices n and a probability p the method returns a networkx graph
    generated according to the Gnp random model,
    i.e. an edge is added between every pair of vertices with probability p

    :param n: The number of vertices in the resulting graph
    :param p: An edge probability
    :return: A networkx graph G generated according to the Gnp random model
    """

    G = nx.fast_gnp_random_graph(n, p)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1.0
    return G


def generate_Gnp_with_planted_clique(n, p, clique_size):
    """
    Given a number of vertices n, a probability p and a clique size
    the method returns a networkx gnp graph with an additional planted clique
    i.e. from the Gnp graph we select a random subset of vertices of size clique_size
    and we add an edge between every pair of vertices in this subset

    :param n: The number of vertices in the resulting graph
    :param p: An edge probability
    :param clique_size: The size of the planted clique
    :return: A networkx graph G generated according to the Gnp random model with an
    additional planted clique on clique_size vertices
    """

    # Generate the Gnp graph
    G = generate_Gnp_graph(n, p)

    # Select a random subset of vertices
    selected_nodes = random.shuffle(list(G.nodes()))[:clique_size]

    # Plant the clique, i.e. add an edge between every pair of nodes in selected_nodes
    for i in selected_nodes:
        for j in selected_nodes:
            if i != j and G.has_edge(i, j) is False:
                G.add_edge(i, j, weight=1.0)
    return G


def generate_complete_graph(n):
    """
    Given a number of vertices n, the method returns a networkx complete graph
    on n vertices with unit weight

    :param n: The number of vertices in the resulting graph
    :return: A networkx complete graph G on n vertices with unit weight
    """

    G = nx.complete_graph(n)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1.0
    return G


def generate_caveman_graph(n, k):
    """
    Given parameters n and k, the method generates a networkx caveman graph
    consisting in k complete subgraphs (cliques) of size n each,
    such that each pair of cliques is connected by an edge

    :param n: The number of vertices in each clique
    :param k: The number of cliques
    :return: A networkx caveman graph G on k cliques of n vertices each
    """

    G = nx.caveman_graph(k, n)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1.0
    return G


def generate_random_regular_graph(n, d):
    """
    Given parameters n and d, the method generates a random graph on n vertices
    such that every vertex has degree d.

    :param n: The number of vertices in the graph
    :param d: The degree of each vertex
    :return: A networkx random graph G on n vertices of degree d each
    """

    G = nx.random_regular_graph(d, n)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1.0
    return G


def generate_bad_cheeger_example(n, c):
    """
    Given parameters n and c, the method generates a graph G for which the Cheeger cut
    fails to produce the sparsest cut. Formally, the graph is a grid of height=sqrt(n)
    and width=c*sqrt(n) such that every point (x, y) is connected to its four neighbours
    (x +/- 1, y), (x, y+/-1) by an edge. All edges have weight 1 except for the vertical
    edges in the middle of the grid, i.e. (height/2, y) - (height/2+1, y) that have weight
    1/height.

    :param n: The number of vertices in the graph up to the constant factor c
    :param c: The scaling factor (greater than 1) of the width = c * height
    :return: A networkx random graph G that is a hard instance for the Cheeger cut
    """

    G = nx.Graph()

    # Initialise parameters width, height
    height = math.floor(math.sqrt(n))
    if height % 2 == 1:
        height += 1
    width = c * height

    # Initialise graph nodes
    for i in range(height):
        for j in range(width):
            G.add_node((i, j))

    # Add horizontal edges
    for i in range(height):
        for j in range(width - 1):
            G.add_edge((i, j), (i, j+1), weight=1.0)

    # Add vertical edges
    for i in range(height - 1):
        for j in range(width):
            if i == height / 2:
                G.add_edge((i, j), (i+1, j), weight=1.0/height)
            else:
                G.add_edge((i, j), (i+1, j), weight=1.0)

    return G


def generate_SBM_same_probs(k, sizes, inter_prob, intra_prob):
    """
    The method generates a networkx random graph according to the Stochastic Block Model (SBM).
    Formally, the graph consists of k clusters on different of various sizes.
    Every pair of vertices in the the same cluster is connected with probability intra_prob.
    Every pair of vertices in different clusters is connected with probability inter_prob.

    :param k: The number of clusters
    :param sizes: An array of length k containing the sizes of each cluster
    :param inter_prob: The probability of connecting two vertices in each cluster
    :param intra_prob: The probability of connecting two vertices in different clusters
    :return: A networkx random graph G according to the SBM model
    """


    probs = [[0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if i == j:
                probs[i][j] = intra_prob
            else:
                probs[i][j] = inter_prob

    G = nx.stochastic_block_model(sizes, probs)

    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1.0
    return G


def generate_HSBM(k, sizes, prob_matrix):
    """
    The method generates a networkx random graph according to the Hierarchical Stochastic Block Model (HSBM).
    Formally, the graph consists of k clusters on different of various sizes.
    Every pair of vertices in the the same cluster is connected with some large probability p.
    Every pair of vertices in different clusters C_i, C_j is connected with probability q_{i,j} that depends on the
    hierarchical structure of the clusters.

    :param k: The number of clusters
    :param sizes: An array of length k containing the sizes of each cluster
    :param prob_matrix: The probability matrix such that prob_matrix[i, j] is the probability of connecting 2 vertices in
    clusters C_i and C_j
    :return: A networkx random graph G according to the HSBM model
    """

    G = nx.stochastic_block_model(sizes, prob_matrix)

    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1.0
    return G


def generate_SBM_with_planted_clique(sizes, probs, c_p):
    G = nx.stochastic_block_model(sizes, probs)
    all_vertices = list(G.nodes())


    first_cluster = all_vertices[:sizes[0]]
    second_cluster = all_vertices[sizes[0]:(sizes[0] + sizes[1])]
    third_cluster = all_vertices[(sizes[0] + sizes[1]): (sizes[0] + sizes[1] + sizes[2])]
    # Plant the cliques

    # First cluster
    random.shuffle(first_cluster)
    selected_nodes = first_cluster[:(int(len(first_cluster) * c_p))]
    for i in selected_nodes:
        for j in selected_nodes:
            if i != j and G.has_edge(i, j) is False:
                G.add_edge(i, j, weight=1.0)

    # Second cluster
    random.shuffle(second_cluster)
    selected_nodes = second_cluster[:(int(len(second_cluster) * c_p))]
    for i in selected_nodes:
        for j in selected_nodes:
            if i != j and G.has_edge(i, j) is False:
                G.add_edge(i, j, weight=1.0)

    # Third cluster
    random.shuffle(third_cluster)
    selected_nodes = third_cluster[:(int(len(third_cluster) * c_p))]
    for i in selected_nodes:
        for j in selected_nodes:
            if i != j and G.has_edge(i, j) is False:
                G.add_edge(i, j, weight=1.0)

    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1.0
    return G
