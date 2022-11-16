"""Guarantees for Spectral Clustering with Fairness Constraints (Kleindessner et al, ICML 2019)"""

import networkx as nx
import numpy as np
from scipy.linalg import null_space
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import check_array

from .base import FairClustering


def get_Vs(array):
    """Get Vs as defined in the paper by Kleindessner et al (ICML 2019)""" 
    counter = 0
    for i in range(len(array)):
        if array[i] == 1:
            counter += 1
    return counter

def get_N(array): 
    """Helper function to get length of column"""
    return len(array)

def n_ones_vector(array): 
    """Helper function to get an array of ones with length of column"""
    return np.ones(len(array))

def build_matrix_F(G, labels):
    """Build the matrix F as defined in the paper by Kleindessner et al (ICML 2019)"""
    num_samples = len(G) 
    F = []
    
    for i in range(max(labels)+1):
        column = [0] * len(G)
        for y in range(len(labels)):
            if labels[y] == i:
                column[y] = 1
        column = np.asarray(column - ((get_Vs(column) / get_N(column)) * n_ones_vector(column)))
        F.append(column)
            
    return np.transpose((np.asarray(F)))


def unnormalized_fair_spectral(G, n_clusters, groups, random_state):
    """Compute the unnormalized fair spectral clustering on graph G for n_clusters # of clusters"""
    laplacian_matrix = nx.laplacian_matrix(G)
    F = build_matrix_F(G, groups)
    Z = null_space(np.transpose(F))
    
    LZ = np.matmul(laplacian_matrix.toarray(), Z)
    TZ = np.transpose(Z)
    fed_matrix = np.matmul(TZ, LZ)
    
    e, v = np.linalg.eigh(fed_matrix)
    
    v, e = v[:,np.argsort(e)], e[np.argsort(e)]
    km = KMeans(init='k-means++', n_clusters=n_clusters, random_state=random_state)
    
    H = np.matmul(Z, v[:, :n_clusters])
    km.fit(H)
    return km.labels_, km.inertia_



class FairSpectral(FairClustering):
    
    def __init__(self, n_clusters=2, num_neighbors=3, metric_str = 'euclidean', random_state=None):
        """Initialize the Fair Spectral clustering method."""
        self.num_neighbors = num_neighbors
        self.metric_str = metric_str
        self.n_clusters = n_clusters
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.labels_ = None
        self.clustering_cost_ = None



    def fit(self, X, s):
        """Fit the method to the dataset and sensitive attribute provided."""
        X = check_array(X)

        if not isinstance(s, (np.ndarray)):
            raise Exception("Protected groups and sensitive attributes must be input as a 1D numpy array.")
        
        s = s.tolist()
        if not isinstance(s, list) or isinstance(s[0], list):
            raise Exception("For this method `s` must be a non-nested list.")

        X, s = np.array(X), np.array(s)
        A = kneighbors_graph(X, self.num_neighbors, metric=self.metric_str, mode='connectivity', p=2, metric_params=None, include_self=False).toarray()
        G = nx.from_numpy_array(A)
        G.edges(data=True)
        self.labels_, self.clustering_cost_ = unnormalized_fair_spectral(G, self.n_clusters, s, self.random_state)
