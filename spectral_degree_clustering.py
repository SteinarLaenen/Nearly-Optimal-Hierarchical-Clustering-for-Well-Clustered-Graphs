import graph_methods as graph
from Tree_Construction import tree
import networkx as nx
import scipy as sp
import numpy as np
import prune_merge
from queue import PriorityQueue
from itertools import chain, combinations
from sklearn.cluster import KMeans

# setting parameter beta with k=2 and gamma = 2
beta = 2**(10*2)


def spectral_degree_clustering(G, k, bucketing=True):
    """
    The implementation of the SpecWRSC Algorithm on input parameters G and k.

    :param G: A networkx graph
    :param k: A target number of clusters
    :return: A Hierarchical Clustering tree T of the graph G
    """
    n = len(list(G.nodes))

    # do spectral clustering
    laplacian_matrix = nx.normalized_laplacian_matrix(G, weight='weight', nodelist=G.nodes)
    eigvals, eigvects = sp.sparse.linalg.eigsh(laplacian_matrix, k=k,
                                            which='SM', return_eigenvectors=True)
    idx = np.argsort(np.real(eigvals))[:k]

    # compute degrees of each vertex and normalize
    degrees = list(G.degree)
    degrees_sort = np.array([i[1] for i in degrees])

    # normalize vector
    W = eigvects[:,idx]/np.sqrt(degrees_sort[:,None])
    kmeans = KMeans(n_clusters=k).fit(W)

    labels = kmeans.labels_

    # get appropriate node labels for the clusters
    clusters = []
    for i in range(k):
        current_cluster_idx = list(np.where(labels == i)[0])
        #renaming the indices using list comprehension
        cluster_nodes = [list(G.nodes)[i] for i in current_cluster_idx]
        clusters.append(cluster_nodes)

    # do the bucketing step
    if bucketing:
        clusters = degree_bucketing(clusters, G)

    # compute the contracted graph with corresponding vertex weights
    contracted_graph = compute_contracted_graph(G, clusters)
    vertex_weights = [len(cluster) for cluster in clusters]


    # construct the final tree
    final_tree = WRSC(G, contracted_graph, vertex_weights, clusters, list(range(len(clusters))))

    cost = compute_cost(final_tree)

    return final_tree, cost

def balanced_f2(nodes, G):
    """
    Takes a graph G as input and subset of nodes inside, and outputs a balanced
    HC tree on the ordering of the nodes based on the Fiedler vector of the induced
    subgraph

    Parameters
    ----------
    G : A networkx undirected graph
        The graph to be clustered hierarchically

    nodes : list of indices which are nodes of G
    Returns
    -------
    list
        A list of lists, where each sublist is a HC of the tree
    """
    n = len(G.nodes)
    if n <= 2:
        tree = recursive_balanced(nodes, G)
    else:
        # compute laplacian and get second eigenvector
        laplacian_matrix = nx.normalized_laplacian_matrix(G, weight='weight',
                                                          nodelist=nodes)
        eigvals, eigvects = sp.sparse.linalg.eigsh(laplacian_matrix, k=2,
                                                   which='SM',
                                                   return_eigenvectors=True)
        idx = np.argsort(np.real(eigvals))

        # compute degrees and normalize the vector by degrees to obtain Fiedler
        # vector
        degrees = list(G.degree)
        degrees_sort = np.array([i[1] for i in degrees])

        f_2 = eigvects[:, idx[1]]/np.sqrt(degrees_sort)
        sort_f2 = np.argsort(f_2)


        reordered_nodes = [nodes[i] for i in sort_f2]

        tree = recursive_balanced(reordered_nodes, G)

    return tree

def degree_bucketing(clusters, G):
    new_clusters = []
    vertex_degrees = list(G.degree)
    for cluster in clusters:
        vertices_in_cluster_with_degree = G.degree(cluster)
        vertex_degrees.sort(key=lambda x: x[1])
        d_min = vertex_degrees[0][1]
        d_max = vertex_degrees[-1][1]

        bucket_intervals = [[d_min, beta*d_min]]
        # construct buckets intervals
        while bucket_intervals[-1][1] < d_max:
            add_interval = [bucket_intervals[-1][1], bucket_intervals[-1][1]*beta]
            bucket_intervals.append(add_interval)


        # create actual buckets
        if len(bucket_intervals) == 1:
            new_cluster = cluster
            new_clusters.append(new_cluster)
        else:
            for bucket_interval in bucket_intervals:
                new_cluster = []
                for vertex in vertices_in_cluster_with_degree:
                    if vertex[1] >= bucket_interval[0] and vertex[1] < bucket_interval[1]:
                        new_cluster.append(vertex[0])

                new_clusters.append(new_cluster)
                print(len(set(new_cluster).intersection(set(cluster))))

    return new_clusters

def compute_contracted_graph(G, clusters):
    total_clusters = len(clusters)
    clusters_dic = {}
    # create label for each cluster
    for i, cluster in enumerate(clusters):
        clusters_dic[i] = cluster

    # compute the adjacency weight matrix
    pairwise_weights = np.zeros((total_clusters, total_clusters))

    for i in range(len(clusters)):
        for j in range(i, len(clusters)):
            if i != j:
                cut = cut_value(G, clusters_dic[i], clusters_dic[j])
                pairwise_weights[i,j] = cut

    return pairwise_weights + pairwise_weights.T

def WRSC(G, contracted_graph, vertex_weights, clusters, subset):
    """
    weighted recursive sparsest cut algorithm
    """
    if type(subset) == int:
        return []
    n = len(subset)



    if n == 1:
        # return balanced tree based on f2 embedding
        return balanced_f2(clusters[subset[0]], nx.subgraph(G, clusters[subset[0]]))
    else:
        split, total_weight, total_size = compute_sparsest_cut(contracted_graph, vertex_weights, subset)
        A = split[0]
        B = split[1]

        cost = total_weight*total_size

        to_return = [WRSC(G, contracted_graph, vertex_weights, clusters, A),
                     WRSC(G, contracted_graph, vertex_weights, clusters, B),
                     float(cost)]
        return to_return

def compute_sparsest_cut(contracted_graph, vertex_weights, subset):
    total_vertices = len(subset)
    total_possible_cuts = list(powerset(subset))

    min_cut = np.inf

    min_cut_sets = [subset[0], subset[1:]]
    min_cut_weight = 0
    min_total_size = 0

    for cut in total_possible_cuts:
        # don't check the trivial cuts
        if len(cut) != 0 and len(cut) != total_vertices:
            cluster_1 = list(cut)
            cluster_2 = []
            for i in subset:
                if i not in cluster_1:
                    cluster_2.append(i)

            total_weight = 0
            for i in cluster_1:
                for j in cluster_2:
                    total_weight += contracted_graph[i,j]


            total_vertex_weight_cluster_1 = sum([vertex_weights[i] for i in cluster_1])
            total_vertex_weight_cluster_2 = sum([vertex_weights[i] for i in cluster_2])

            # store total size of the cut
            total_size = total_vertex_weight_cluster_1 + total_vertex_weight_cluster_2

            # compute the sparsest cut
            sparsest_cut = total_weight/(total_vertex_weight_cluster_1*total_vertex_weight_cluster_2)

            if sparsest_cut < min_cut:
                min_cut_sets = [cluster_1, cluster_2]
                min_cut_weight = total_weight
                min_total_size = total_size


    return min_cut_sets, min_cut_weight, min_total_size

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def recursive_balanced(nodes, G):
    """
    Takes a graph G as input with nodes ordered by some metric, and outputs a
    HC of the tree which splits the tree in two recursively so it is balanced

    Parameters
    ----------
    nodes : list
        A list of nodes in the graph G

    Returns
    -------
    list
        A list of lists, where each sublist is a HC of the tree. Appended to
        each list is a float64 which is the cost of that level.
    """
    n = len(nodes)

    if n == 1:
        return nodes
    else:
        r = int(n/2)
        A = nodes[:r]
        B = nodes[r:]

        cut = cut_value(G, A, B)
        cost = cut*len(nodes)


        return [recursive_balanced(A, G), recursive_balanced(B, G), float(cost)]

def compute_cost(tree):
    """
    Easy function to compute the cost of the tree. First, flatten list of lists
    (which is how the tree is represented). Then only sum up the cut*size values
    which are floats instead of integers
    """
    flattened = flatten(tree)
    cost = sum([i for i in flattened if type(i)==np.float64 or type(i)==float])
    return cost

def flatten(l, ltypes=(list, tuple)):
    """
    Source: From http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html

    flatten(l[, ltypes]) -> list

    Flattens a list.

    Parameters
    ----------
    l : list
        The list to be flattened.
    ltypes : tuple of types, optional
        The types of lists to flatten.

    Returns
    -------
    list
        The flattened list.

    Examples
    --------
    >>> flatten([[1, 2], [3, 4]])
    [1, 2, 3, 4]
    >>> flatten([[1, 2], [3, [4]]])
    [1, 2, 3, [4]]
    >>> flatten([[1, 2], [3, [4]]], ltypes=(list, tuple))
    [1, 2, 3, 4]
    """

    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def cut_value(G, S, T):
    """
    Given two sets of vertices S and T, we compute and return the cut value between S and T,
    i.e. the sum of the weights of edges with one endpoint in S and the other in T,
    sometimes denoted as w(S, T).

    :param G: A networkx graph
    :param S: A list of vertices in G
    :param T: A list of vertices in G
    :return: The cut value between S and T in G
    """

    # Deal with the corner cases
    if nx.is_empty(G) or len(S) == 0 or len(T) == 0:
        return 0

    # cut_val stores the overall cut value
    cut_val = 0

    # If S and T have small sizes, we compute the cut by looping through S and T
    if G.number_of_nodes() ** 1.5 > len(S) * len(T):
        for u in S:
            for v in T:
                if G.has_edge(u, v) and 'weight' in G.edges[u, v]:
                    cut_val += G[u][v]['weight']
                elif G.has_edge(u, v):
                    cut_val += 1.0
    # If S or T has large size, we compute the cut by looping through the edges in G
    else:
        vertices_in_S_or_T = {}
        for vertex in S:
            vertices_in_S_or_T[vertex] = 'S'
        for vertex in T:
            vertices_in_S_or_T[vertex] = 'T'
        for edge in list(G.edges()):
            if edge[0] in vertices_in_S_or_T and edge[1] in vertices_in_S_or_T:
                if vertices_in_S_or_T[edge[0]] != vertices_in_S_or_T[edge[1]]:
                    u = edge[0]
                    v = edge[1]
                    if G.has_edge(u, v) and 'weight' in G.edges[u, v]:
                        cut_val += G[u][v]['weight']
                    elif G.has_edge(u, v):
                        cut_val += 1.0
                    #cut_val += G[edge[0]][edge[1]]['weight']
    return cut_val
