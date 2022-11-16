"""Fair Clustering Through Fairlets by (Chierichetti et al, NeurIPS 2017)"""

import numpy as np
from sklearn.utils import check_array
from math import gcd
import random

from .base import FairClustering

def kcenters(X, n_clusters):
    """Greedy K-Center implementation, modified slightly for easy use with fairlet decomposition."""
    num_samples = len(X)
    centers = [int(np.random.randint(0, num_samples, 1))]
    costs = []

    while True:
        data = list(set(range(0, num_samples)) - set(centers))
        potential_center = [(i, min([np.linalg.norm(X[i] - X[j]) for j in centers])) for i in data]
        potential_center = sorted(potential_center, key=lambda x: x[1], reverse=True)
        costs.append(potential_center[0][1])
        if len(centers) < n_clusters:
            centers.append(potential_center[0][0])
        else:
            break

    return centers


def check_balance(R, B, alpha, beta):
    """Find whether or not we can have a balanced solution given the current ALPHA and BETA."""
    if R == 0 and B == 0:
        return True
    if R == 0 or  B == 0:
        return False
    return min(float(R/B), float(B/R)) >= float(beta/alpha)


def compute_fairlet_cost(X, fairlet):
    """Given set of fairlets, compute the centers and associated costs."""
    centers, costs = [], []
    costs = [(idx_i, max([np.linalg.norm(X[idx_i] - X[idx_j]) for idx_j in fairlet])) for idx_i in fairlet]
    costs = sorted(costs, key=lambda x:x[1], reverse=False)
    center, cost = costs[0][0], costs[0][1]
    centers.append(center)
    costs.append(cost)
    return centers, costs


def compute_decomposition(X, s_R, s_B, alpha, beta):
    """Compute vanilla fairlet decomposition according to Chierichetti et al (NeurIPS 2017). This is not the optimal MCMF solution as that runs too slow for practical use."""
    fairlets, fairlet_centers, fairlet_costs = [], [], []
    R,B = len(s_R), len(s_B)
    if R == 0 and B == 0:
        return fairlets, fairlet_centers, fairlet_costs

    b,r = 0,0
    np.random.shuffle(s_R)
    np.random.shuffle(s_B)

    while ((R - r) - (B - b)) >= (alpha - beta) and (R - r) >= alpha and (B - b) >= beta:
        fairlet = s_R[r: (r + alpha)] + s_B[b: (b + beta)]
        fairlets.append(fairlet)
        centers, costs = compute_fairlet_cost(X, fairlet)
        fairlet_centers.append(centers)
        fairlet_costs.append(costs)
        r += alpha
        b += beta
        
    if ((R - r) + (B - b)) >= 1 and ((R - r) + (B - b)) <= (beta + alpha):
        fairlet = s_R[r:] + s_B[b:]
        fairlets.append(fairlet)
        centers, costs = compute_fairlet_cost(X, fairlet)
        fairlet_centers.append(centers)
        fairlet_costs.append(costs)
        r = R
        b = B
    elif ((R - r) != (B - b)) and ((B - b) >= beta):
        fairlet = s_R[r: r + (R - r) - (B - b) + beta] + s_B[b: (b + beta)]
        fairlets.append(fairlet)
        centers, costs = compute_fairlet_cost(X, fairlet)
        fairlet_centers.append(centers)
        fairlet_costs.append(costs)
        r += (R - r) - (B - b) + beta
        b += beta

    if (R-r) != (B-b):
        raise Exception("Error in fairlet decomposition.")

    for i in range(R - r):
        fairlet = [s_R[r + i], s_B[b + i]]
        fairlets.append(fairlet)
        centers, costs = compute_fairlet_cost(X, fairlet)
        fairlet_centers.append(centers)
        fairlet_costs.append(costs)

    return fairlets, fairlet_centers


def compute_mapping(X, centers):
    """Helper function to compute a sample to center mapping given a set of centers"""
    num_samples = len(X)
    mapping = [(i, sorted([(j, np.linalg.norm(X[i] - X[j])) for j in centers], key=lambda x: x[1], reverse=False)[0][0]) for i in range(num_samples)]
    return mapping


def compute_clustering(X, n_clusters, fairlets, fairlet_centers):
    """Compute clustering labels and costs from fairlets"""
    final_cluster_assignments = []
    final_cluster_centers = []
    final_costs = []

    for num_cluster in range(1, min(n_clusters+1, len(fairlet_centers)), 1):
        fairlet_data = [X[i] for i in fairlet_centers]
        centers = kcenters(fairlet_data, num_cluster)
        mapping = compute_mapping(fairlet_data, centers)

        final_clusters = []
        for fairlet_id, final_cluster in mapping:
            for sample in fairlets[fairlet_id]:
                final_clusters.append((sample, fairlet_centers[final_cluster]))
                
        final_centers = [fairlet_centers[i] for i in centers]
        final_cluster_assignments = final_clusters
        final_cluster_centers = final_centers
        final_costs = max([min([np.linalg.norm(X[j] - i) for j in final_cluster_centers]) for i in X])

    cluster_idx_id = {cluster_idx[0]: cluster_id for cluster_idx, cluster_id in zip(final_cluster_centers, range(n_clusters))}
    labels = []
    for idx, cluster_idx in final_cluster_assignments:
        cluster_id = cluster_idx_id[cluster_idx[0]]
        labels.append(cluster_id)

    return labels, final_costs


class FairletDecomposition(FairClustering):
    
    def __init__(self, n_clusters=2, alpha=None, beta=None, random_state=None):
        """Initialize the fairlet decomposition method."""
        
        if alpha is None or beta is None:
            raise Exception("For this method, ALPHA and BETA must be provided for the two protected groups.")
        elif gcd(alpha, beta) != 1:
            raise Exception("For this method, GCD of ALPHA and BETA must be 1.")

        if beta > alpha:
            alpha, beta = beta, alpha

        self.alpha = alpha
        self.beta = beta
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

        if np.max(np.unique(s)) > 1:
            raise Exception("For this method `s` can only have two protected groups.")

        X, s = np.array(X), np.array(s)

        s_R = np.where(s == 0)[0].tolist()
        s_B = np.where(s == 1)[0].tolist()
        R, B = len(s_R), len(s_B) 

        if R < B:
            s_R, s_B = s_B, s_R
            R, B = B, R

        if not check_balance(R, B, self.alpha, self.beta):
            raise Exception("Provided ALPHA and BETA values are not feasible with `s`.")

        fairlets, fairlet_centers = compute_decomposition(X, s_R, s_B, self.alpha, self.beta)
        self.labels_, self.clustering_cost_ = compute_clustering(X, self.n_clusters, fairlets, fairlet_centers)

