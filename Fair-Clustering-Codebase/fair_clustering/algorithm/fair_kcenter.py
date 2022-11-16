""" KFC: A Scalable Approximation Algorithm for k-center Fair Clustering (Harb et al, NeurIPS 2020) """

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from scipy.spatial import distance_matrix
import numpy as np
from math import ceil, floor
import pulp

from .base import FairClustering


def reformat_list(X, s):
    """If provided list as input this function reformats it to a dict"""
    num_groups = np.max(np.unique(s)) + 1
    s_dict = {i: [] for i in range(num_groups)}
    for idx in range(X.shape[0]):
        s_dict[s[idx]].append(idx)
    return s_dict


def k_center(X, n_clusters):
    """Greedy K-Center Implementation"""

    num_samples = X.shape[0]
    centers = [np.random.randint(0, num_samples)]
    num_centers = len(centers)

    while num_centers < n_clusters:
        farthest_dist = float("-inf")
        farthest_center = -1

        dist_mat = distance_matrix(X, X[centers])

        for idx in range(num_samples):
            curr_dist = dist_mat[idx].min()
            if curr_dist > farthest_dist:
                farthest_dist = curr_dist
                farthest_center = idx

        centers.append(farthest_center)
        num_centers = len(centers)

    return centers


def get_proportions(X, n_clusters, s, delta):
    """Derive ALPHA and BETA from DELTA as in Bera et al's approach (NeurIPS 2019)"""
    num_groups = len(s)
    num_samples = X.shape[0]
    alpha = {i: 0 for i in range(num_groups)}
    beta = {i: 0 for i in range(num_groups)}

    for group_id in range(num_groups):
        group_size = len(s[group_id])
        const = group_size / num_samples
        alpha[group_id] = const / (1 - delta)
        beta[group_id] = const * (1 - delta)

    return (alpha, beta)


def samples_to_groups(X, s):
    """Reformulate protected group membership to a format where sample indices of `X` are keys in dict and groups are values"""
    num_groups = len(s)
    n_samples = X.shape[0]
    samples_groups_dict = {i: [] for i in range(n_samples)}
    for group_id in range(num_groups):
        for sample in s[group_id]:
            samples_groups_dict[sample].append(group_id)

    return samples_groups_dict


def calculate_mav(X, n_clusters, s, cluster_labels, alpha, beta):
    """Calculate the maximum additive violation according to Harb et al (NeurIPS 2020) / Bera et al (NeurIPS 2019)"""
    num_groups = len(s)
    samples_groups_dict = samples_to_groups(X, s)
    mav = 0

    for cluster_id in range(n_clusters):
        cluster_elems = np.where(cluster_labels == cluster_id)[0].tolist()
        cluster_size = len(cluster_elems)
        for group_id in range(num_groups):
            count = 0
            for sample in cluster_elems:
                if group_id in samples_groups_dict[sample]:
                    count += 1

            upper_bound = alpha[group_id] * cluster_size
            lower_bound = beta[group_id] * cluster_size
            if count > upper_bound:
                mav = max(mav, ceil(count - upper_bound))
            elif count < lower_bound:
                mav = max(mav, ceil(lower_bound - count))

    return mav


def compute_labels(cluster_dict, n_clusters, X):
    """Compute labels from a dict representing clustering assignment"""
    num_samples = X.shape[0]
    cluster_labels = [float("-inf")] * num_samples

    for cluster_id in range(n_clusters):
        for idx in cluster_dict[cluster_id]:
            cluster_labels[idx] = cluster_id

    return cluster_labels


def compute_clustering_cost(cluster_dict, n_clusters, X, centers):
    cost = max([distance_matrix([centers[cluster_id]], X[cluster_dict[cluster_id]]).max() if len(
        cluster_dict[cluster_id]) > 0 else 0 for cluster_id in range(n_clusters)])
    return cost


def FQ_LP(X, centers, n_clusters, s, alpha, beta, lambda_, dist_centers, solver):
    """The frequency distributor linear program defined in Harb et al (NeurIPS 2020)"""
    samples_groups_dict = samples_to_groups(X, s)
    num_samples = X.shape[0]
    joiners = {}

    for idx in range(num_samples):
        connect = []
        for cluster_id in range(n_clusters):
            if dist_centers[idx, cluster_id] <= lambda_:
                connect.append(cluster_id)

        if connect == []:
            # No feasible LP solution possible
            return 0, None

        subset = tuple(connect)
        sig = tuple(samples_groups_dict[idx])

        if subset not in joiners:
            joiners[subset] = {}

        if sig not in joiners[subset]:
            joiners[subset][sig] = []

        joiners[subset][sig].append(idx)

    variables = {}

    for subset in joiners.keys():
        for sig in joiners[subset].keys():
            for cluster_id in subset:
                variable_sig = tuple([tuple(subset), tuple([sig]), tuple([cluster_id])])
                variables[variable_sig] = pulp.LpVariable(str(variable_sig).replace(' ', ''), lowBound=0)

    Lp_prob = pulp.LpProblem('Problem', pulp.LpMaximize)
    Lp_prob += 1

    num_groups = len(s)
    for cluster_id in range(n_clusters):
        cluster_vars = []
        for subset in joiners.keys():
            if cluster_id in subset:
                for sig in joiners[subset].keys():
                    var_sig = tuple([tuple(subset), tuple([sig]), tuple([cluster_id])])
                    cluster_vars.append(variables[var_sig])

        cluster_vars_sum = pulp.lpSum(cluster_vars)

        for group_id in range(num_groups):
            group_vars = []
            for subset in joiners.keys():
                if cluster_id in subset:
                    for sig in joiners[subset].keys():
                        if group_id in sig:
                            var_sig = tuple([tuple(subset), tuple([sig]), tuple([cluster_id])])
                            group_vars.append(variables[var_sig])

            group_vars_sum = pulp.lpSum(group_vars)

            Lp_prob += group_vars_sum <= alpha[group_id] * cluster_vars_sum
            Lp_prob += group_vars_sum >= beta[group_id] * cluster_vars_sum

    for subset in joiners.keys():
        for sig in joiners[subset].keys():
            sig_subset_eq = len(joiners[subset][sig])

            sig_subset_vars = []
            for cluster_id in subset:
                var_sig = tuple([tuple(subset), tuple([sig]), tuple([cluster_id])])
                sig_subset_vars.append(variables[var_sig])

            Lp_prob += pulp.lpSum(sig_subset_vars) == sig_subset_eq
    try:
        status = Lp_prob.solve(solver)
    except:
        return 0, None

    if pulp.LpStatus[status] != 'Optimal':
        return status, None

    clusters = {i: [] for i in range(n_clusters)}

    for subset in joiners.keys():
        for sig in joiners[subset].keys():
            sig_subset_eq = len(joiners[subset][sig])
            probs = [variables[tuple([tuple(subset), tuple([sig]), tuple([cluster_id])])].value() / sig_subset_eq for
                     cluster_id in subset]
            probs = np.array(probs)
            probs[probs < 0] = 0
            draw = np.random.choice(subset, sig_subset_eq, p=probs)
            for i, idx in enumerate(joiners[subset][sig]):
                clusters[draw[i]].append(idx)

    return status, clusters


def kfc(X, n_clusters, s, alpha, beta, tol, solver):
    centers = X.copy()
    centers_idx = k_center(centers, n_clusters)
    centers = centers[centers_idx]

    num_samples = X.shape[0]

    dist_centers = distance_matrix(X, centers)

    L, R, feasible = 0, 2 * np.max(dist_centers), False

    while (R - L) > tol or not feasible:
        lambda_ = (L + R) / 2
        flag = False

        for idx in range(num_samples):
            if dist_centers[idx].min() > lambda_:
                L, feasible, flag = lambda_, False, True
                continue

        if flag:
            continue

        LP_status, clusters = FQ_LP(X, centers, n_clusters, s, alpha, beta, lambda_, dist_centers, solver)

        if pulp.LpStatus[LP_status] == 'Optimal':
            R, feasible = lambda_, True
        else:
            L, feasible = lambda_, False

    return compute_labels(clusters, n_clusters, X), compute_clustering_cost(clusters, n_clusters, X, centers)


class FairKCenter(FairClustering):

    def __init__(self, n_clusters=2, delta=0.1, tol=1e-3, random_state=None, solver_name='CPLEX'):
        """Initialize the KFC Fair K-Center method."""
        if solver_name == 'CPLEX':
            self.solver = pulp.CPLEX_PY(msg=0)
        elif solver_name == 'GUROBI':
            self.solver = pulp.GUROBI_CMD()
        else:
            raise Exception("For this method only CPLEX and GUROBI are supported solvers.")

        if delta is None:
            raise Exception(
                "For this method DELTA must be provided, where DELTA=0 translates to ideal proportion and DELTA=1 translates to no fairness imposed.")

        self.delta = delta
        self.n_clusters = n_clusters
        self.tol = tol
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
        if isinstance(s, list) and not isinstance(s[0], list):
            s = reformat_list(X, s)
        else:
            raise Exception("For this method `s` must be a non-nested list.")
        self.alpha, self.beta = get_proportions(X, self.n_clusters, s, self.delta)
        self.labels_, self.clustering_cost_ = kfc(X, self.n_clusters, s, self.alpha, self.beta, self.tol, self.solver)
