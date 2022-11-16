from abc import ABC, abstractmethod
from sklearn.utils.validation import check_is_fitted
import numpy as np


class FairClustering(ABC):
    """
    Abstract base class for all fair clustering algorithms.

    Attributes
    ----------
    labels_: ndarray of shape (n_samples,)
        Labels of each point

    clustering_cost_: float
        Clustering utility/cost for the given clustering algorithm

    """

    @abstractmethod
    def __init__(self, n_clusters: int, ):
        self.labels_ = None

    @abstractmethod
    def fit(self, X: np.ndarray, s: np.ndarray):
        """
        Fit the clustering algorithm.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input samples.

        s : np.ndarray of shape (n_samples,)
            The sensitive attributes.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        pass

    def fit_predict(self, X: np.ndarray, s: np.ndarray):
        """
        Perform fair clustering on dataset `X` and sensitive attribute `s` then return cluster labels.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input samples.
             
        s: np.ndarray of shape (n_samples,)
            The sensitive attributes.


        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype=np.int64
            Cluster labels.
        """

        self.fit(X, s)
        return self.labels_
    
    
    def predict(self, X: np.ndarray, s: np.ndarray):
        """
         Predict cluster labels on dataset `X` and sensitive attribute `s` for the given clustering model.
         Parameters
         ----------
         X : np.ndarray of shape (n_samples, n_features)
             The input samples.
         s: np.ndarray of shape (n_samples,)
             The sensitive attributes.
         Returns
         -------
         labels : ndarray of shape (n_samples,), dtype=np.int64
             Cluster labels.
         """

        check_is_fitted(self, ['clustering_cost_', 'labels_'])
        raise NotImplementedError
