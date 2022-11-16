"""Scalable Fair Clustering by (Backurs et al, ICML 2019)"""

import numpy as np
from collections import defaultdict
import kmedoids
from sklearn.metrics.pairwise import euclidean_distances
from math import gcd
from sklearn.utils import check_array
import random

from .base import FairClustering

class TreeNode:
 
    def __init__(self):
        self.children = []
 
    def set_cluster(self, cluster):
        self.cluster = cluster
 
    def add_child(self, child):
        self.children.append(child)
 
    def populate_colors(self, colors):
        """Populate auxiliary lists of red and blue points for each node, bottom-up"""
        self.reds = []
        self.blues = []
        if len(self.children) == 0:
            # Leaf
            for i in self.cluster:
                if colors[i] == 0:
                    self.reds.append(i)
                else:
                    self.blues.append(i)
        else:
            # Not a leaf
            for child in self.children:
                child.populate_colors(colors)
                self.reds.extend(child.reds)
                self.blues.extend(child.blues)


class ScalableFairletDecomposition(FairClustering):

    def __init__(self, n_clusters=2, epsilon=0.0001, alpha=None, beta=None, random_state=None):
        """Initialize the Scalable Fair Clustering (ICML 2019) algorithm"""
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

        self.epsilon = epsilon
        self.fairlets = []
        self.fairlet_centers = []

        self.labels_ = None
        self.clustering_cost_ = None


    def kmedian_cost(self, centroids, X):
        """Computes and returns k-median cost for given X and centroids"""
        return sum(np.amin(np.concatenate([np.linalg.norm(X[:,:]-X[centroid,:], axis=1).reshape((X.shape[0], 1)) for centroid in centroids], axis=1), axis=1))

    def fair_kmedian_cost(self, centroids, X):
        """Return the fair k-median cost for given centroids and fairlet decomposition"""
        total_cost = 0
        for i in range(len(self.fairlets)):
            # Choose index of centroid which is closest to the i-th fairlet center
            cost_list = [np.linalg.norm(X[centroids[j],:]-X[self.fairlet_centers[i],:]) for j in range(len(centroids))]
            cost, j = min((cost, j) for (j, cost) in enumerate(cost_list))
            # Assign all points in i-th fairlet to above centroid and compute cost
            total_cost += sum([np.linalg.norm(X[centroids[j],:]-X[point,:]) for point in self.fairlets[i]])
        return total_cost


    def balanced(self, p, q, r, b):
        if r==0 and b==0:
            return True
        if r==0 or b==0:
            return False
        return min(r*1./b, b*1./r) >= p*1./q

    def make_fairlet(self, points, X):
        """Adds fairlet to fairlet decomposition, returns median cost"""
        self.fairlets.append(points)
        cost_list = [sum([np.linalg.norm(X[center,:]-X[point,:]) for point in points]) for center in points]
        cost, center = min((cost, center) for (center, cost) in enumerate(cost_list))
        self.fairlet_centers.append(points[center])
        return cost


    def basic_fairlet_decomposition(self, p, q, blues, reds, X):
        """
        Computes vanilla (p,q)-fairlet decomposition of given points (Lemma 3 in NIPS17 paper).
        Returns cost.
        Input: Balance parameters p,q which are non-negative integers satisfying p<=q and gcd(p,q)=1.
        "blues" and "reds" are sets of points indices with balance at least p/q.
        """
        assert p <= q, "Please use balance parameters in the correct order"
        if len(reds) < len(blues):
            temp = blues
            blues = reds
            reds = temp
        R = len(reds)
        B = len(blues)
        assert self.balanced(p, q, R, B), "Input sets are unbalanced: "+str(R)+","+str(B)
 
        if R==0 and B==0:
            return 0
 
        b0 = 0
        r0 = 0
        cost = 0
        while (R-r0)-(B-b0) >= q-p and R-r0 >= q and B-b0 >= p:
            temp_cost = self.make_fairlet(reds[r0:r0+q]+blues[b0:b0+p], X)
            cost += temp_cost
            r0 += q
            b0 += p
        if R-r0 + B-b0 >=1 and R-r0 + B-b0 <= p+q:
            temp_cost = self.make_fairlet(reds[r0:]+blues[b0:], X)
            cost += temp_cost
            r0 = R
            b0 = B
        elif R-r0 != B-b0 and B-b0 >= p:
            temp_cost = self.make_fairlet(reds[r0:r0+(R-r0)-(B-b0)+p]+blues[b0:b0+p], X)
            cost += temp_cost
            r0 += (R-r0)-(B-b0)+p
            b0 += p
        assert R-r0 == B-b0, "Error in computing fairlet decomposition"
        for i in range(R-r0):
            temp_cost = self.make_fairlet([reds[r0+i], blues[b0+i]], X)
            cost += temp_cost
        return cost



    def node_fairlet_decomposition(self, p, q, node, X, donelist, depth):
        # Leaf                                                                                          
        if len(node.children) == 0:
            node.reds = [i for i in node.reds if donelist[i]==0]
            node.blues = [i for i in node.blues if donelist[i]==0]
            assert self.balanced(p, q, len(node.reds), len(node.blues)), "Reached unbalanced leaf"
            ret_cost = self.basic_fairlet_decomposition(p, q, node.blues, node.reds, X)
            return ret_cost 
 
        # Preprocess children nodes to get rid of points that have already been clustered
        for child in node.children:
            child.reds = [i for i in child.reds if donelist[i]==0]
            child.blues = [i for i in child.blues if donelist[i]==0]
 
        R = [len(child.reds) for child in node.children]
        B = [len(child.blues) for child in node.children]
 
        if sum(R) == 0 or sum(B) == 0:
            assert sum(R)==0 and sum(B)==0, "One color class became empty for this node while the other did not"
            return 0
 
        NR = 0
        NB = 0
 
        # Phase 1: Add must-remove nodes
        for i in range(len(node.children)):
            if R[i] >= B[i]:
                must_remove_red = max(0, R[i] - int(np.floor(B[i]*q*1./p)))
                R[i] -= must_remove_red
                NR += must_remove_red
            else:
                must_remove_blue = max(0, B[i] - int(np.floor(R[i]*q*1./p)))
                B[i] -= must_remove_blue
                NB += must_remove_blue
 
        # Calculate how many points need to be added to smaller class until balance
        if NR >= NB:
            # Number of missing blues in (NR,NB)
            missing = max(0, int(np.ceil(NR*p*1./q)) - NB)
        else:
            # Number of missing reds in (NR,NB)
            missing = max(0, int(np.ceil(NB*p*1./q)) - NR)
         
        # Phase 2: Add may-remove nodes until (NR,NB) is balanced or until no more such nodes
        for i in range(len(node.children)):
            if missing == 0:
                assert self.balanced(p, q, NR, NB), "Something went wrong"
                break
            if NR >= NB:
                may_remove_blue = B[i] - int(np.ceil(R[i]*p*1./q))
                remove_blue = min(may_remove_blue, missing)
                B[i] -= remove_blue
                NB += remove_blue
                missing -= remove_blue
            else:
                may_remove_red = R[i] - int(np.ceil(B[i]*p*1./q))
                remove_red = min(may_remove_red, missing)
                R[i] -= remove_red
                NR += remove_red
                missing -= remove_red
 
        # Phase 3: Add unsaturated fairlets until (NR,NB) is balanced
        for i in range(len(node.children)):
            if self.balanced(p, q, NR, NB):
                break
            if R[i] >= B[i]:
                num_saturated_fairlets = int(R[i]/q)
                excess_red = R[i] - q*num_saturated_fairlets
                excess_blue = B[i] - p*num_saturated_fairlets
            else:
                num_saturated_fairlets = int(B[i]/q)
                excess_red = R[i] - p*num_saturated_fairlets
                excess_blue = B[i] - q*num_saturated_fairlets
            R[i] -= excess_red
            NR += excess_red
            B[i] -= excess_blue
            NB += excess_blue
 
        assert self.balanced(p, q, NR, NB), "Constructed node sets are unbalanced"
 
        reds = []
        blues = []
        for i in range(len(node.children)):
            for j in node.children[i].reds[R[i]:]:
                reds.append(j)
                donelist[j] = 1
            for j in node.children[i].blues[B[i]:]:
                blues.append(j)
                donelist[j] = 1
 
        assert len(reds)==NR and len(blues)==NB, "Something went horribly wrong"
 
        return self.basic_fairlet_decomposition(p, q, blues, reds, X) + sum([self.node_fairlet_decomposition(p, q, child, X, donelist, depth+1) for child in node.children])

    def tree_fairlet_decomposition(self, p, q, root, X, colors):
        "Main fairlet clustering function, returns cost wrt original metric (not tree metric)"
        assert p <= q, "Please use balance parameters in the correct order"
        root.populate_colors(colors)
        assert self.balanced(p, q, len(root.reds), len(root.blues)), "Dataset is unbalanced"
        root.populate_colors(colors)
        donelist = [0] * X.shape[0]
        return self.node_fairlet_decomposition(p, q, root, X, donelist, 0)

    def build_quadtree(self, X, max_levels=0, random_shift=True):
        """If max_levels=0 there no level limit, quadtree will partition until all clusters are singletons"""
        dimension = X.shape[1]
        lower = np.amin(X, axis=0)
        upper = np.amax(X, axis=0)
 
        shift = np.zeros(dimension)
        if random_shift:
            for d in range(dimension):
                spread = upper[d] - lower[d]
                shift[d] = np.random.uniform(0, spread)
                upper[d] += spread
 
        return self.build_quadtree_aux(X, range(X.shape[0]), lower, upper, max_levels, shift)
     

    def build_quadtree_aux(self, X, cluster, lower, upper, max_levels, shift):
        """
        "lower" is the "bottom-left" (in all dimensions) corner of current hypercube
        "upper" is the "upper-right" (in all dimensions) corner of current hypercube
        """
 
        dimension = X.shape[1]
        cell_too_small = True
        for i in range(dimension):
            if upper[i]-lower[i] > self.epsilon:
                cell_too_small = False
 
        node = TreeNode()
        if max_levels==1 or len(cluster)<=1 or cell_too_small:
            # Leaf
            node.set_cluster(cluster)
            return node
     
        # Non-leaf
        midpoint = 0.5 * (lower + upper)
        subclusters = defaultdict(list)
        for i in cluster:
            subclusters[tuple([X[i,d]+shift[d]<=midpoint[d] for d in range(dimension)])].append(i)
        for edge, subcluster in subclusters.items():
            sub_lower = np.zeros(dimension)
            sub_upper = np.zeros(dimension)
            for d in range(dimension):
                if edge[d]:
                    sub_lower[d] = lower[d]
                    sub_upper[d] = midpoint[d]
                else:
                    sub_lower[d] = midpoint[d]
                    sub_upper[d] = upper[d]
            node.add_child(self.build_quadtree_aux(X, subcluster, sub_lower, sub_upper, max_levels-1, shift))
        return node



    def scalable_fair_clustering(self, X, s):
        """Main function that combines all the above functions"""
        root = self.build_quadtree(X)
        cost = self.tree_fairlet_decomposition(self.beta, self.alpha, root, X, s)
        fairlet_center_idx = [X[index] for index in self.fairlet_centers]
        fairlet_center_pt = np.array([np.array(xi) for xi in fairlet_center_idx])
        dists = euclidean_distances(fairlet_center_pt)
        kmed_obj = kmedoids.fasterpam(dists, self.n_clusters)
        centroids = [self.fairlet_centers[index] for index in kmed_obj.medoids]
        kmedian_cost = self.fair_kmedian_cost(centroids, X)
        labels = []
        centroids = X[centroids]
        for idx, x in enumerate(X):
            min_label = float('-inf')
            min_dist = float('inf')
            for cluster_label, centroid in enumerate(centroids):
                curr_dist = np.linalg.norm(x - centroid)
                if curr_dist < min_dist:
                    min_dist, min_label = curr_dist, cluster_label
            labels.append(min_label)

        return labels, kmedian_cost


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

        self.labels_, self.clustering_cost_ = self.scalable_fair_clustering(X, s)
