from collections import defaultdict
import numpy as np
import scipy
import pandas as pd
from typing import List
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment


def nmi(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)


def ari(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)


def acc(y_true, y_pred):
    y_true, y_pred = pd.Series(list(y_true)), pd.Series(list(y_pred))
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def balance(labels, X, s):
    data, lbls, groups = X, labels, s

    npoints = data.shape[0]
    num_groups = np.unique(groups).shape[0]

    ideal_proportion = defaultdict(float)
    for g in range(num_groups):
        ideal_proportion[g] = np.count_nonzero(np.array(groups == g))

    for g in range(num_groups):
        ideal_proportion[g] /= float(npoints)

    membership = defaultdict(lambda: defaultdict(float))
    cluster_sizes = defaultdict(float)
    for idx, x in enumerate(data):
        cluster_k = lbls[idx]
        for g in range(num_groups):
            if groups[idx] == g:
                membership[g][cluster_k] += 1.0

        cluster_sizes[cluster_k] += 1.0

    val = float('inf')

    for cluster_k in np.unique(lbls):
        for g in range(num_groups):
            if (membership[g][cluster_k] == 0):
                return 0

            a = (float(membership[g][cluster_k]) / float(cluster_sizes[cluster_k])) / float(ideal_proportion[g])
            b = float(ideal_proportion[g]) / (float(membership[g][cluster_k]) / float(cluster_sizes[cluster_k]))
            val = min(min(a, b), val)

    return val


def entropy(y_pred: np.ndarray, s: np.ndarray, eps=1e-15) -> List[float]:
    entropy_list = []
    for s_val in np.sort(np.unique(s)):
        n_clusters = np.max(y_pred)+1
        group_dis = [0]*(n_clusters)
        cluster_sizes = [0]*(n_clusters)
        for i, cluster_idx in enumerate(y_pred):
            cluster_sizes[cluster_idx] += 1
            if s[i] == s_val:
                group_dis[cluster_idx] += 1
        
        e = 0
        for cluster_idx in range(n_clusters):
            if cluster_sizes[cluster_idx] == 0:
                term = 0
            else:
                term = group_dis[cluster_idx]/cluster_sizes[cluster_idx]
            e += term * np.log(term + eps)  
        e = -e
        entropy_list.append(e)

    return np.mean(entropy_list)
