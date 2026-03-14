from kmodes.kmodes import KModes
from sklearn.base import BaseEstimator, ClusterMixin

from ._bahc import BiasAwareHierarchicalClustering


class BiasAwareHierarchicalKModes(BaseEstimator, ClusterMixin):
    """Bias-Aware Hierarchical k-Modes Clustering.

    Parameters
    ----------
    bahc_max_iter : int
        Maximum number of iterations to run the hierarchical splitting procedure.
    bahc_min_cluster_size : int
        The minimum size a cluster must have to be further split.
    **kmodes_params : dict
        Additional hyperparameters to pass to KModes upon instantiation.

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm.
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point.
        Lower labels correspond to higher discrimination scores.
    scores_ : ndarray of shape (n_clusters_,)
        Discrimination scores for each cluster.

    References
    ----------
    .. [1] J. Misztal-Radecka, B. Indurkhya, "Bias-Aware Hierarchical Clustering
           for detecting the discriminated groups of users in recommendation systems",
           Information Processing & Management, vol. 58, no. 3, May. 2021.

    Examples
    --------
    >>> from unsupervised_bias_detection.cluster import BiasAwareHierarchicalKModes
    >>> import numpy as np
    >>> X = np.array([[0, 1], [0, 2], [0, 0], [1, 4], [1, 5], [1, 3]])
    >>> y = np.array([0, 0, 0, 10, 10, 10])
    >>> bahc = BiasAwareHierarchicalKModes(
    ...     bahc_max_iter=1, bahc_min_cluster_size=1, random_state=12
    ... ).fit(X, y)
    >>> bahc.labels_
    array([0, 0, 0, 1, 1, 1], dtype=uint32)
    >>> bahc.scores_
    array([ 10., -10.])
    """

    def __init__(self, bahc_max_iter, bahc_min_cluster_size, **kmodes_params):
        if "n_clusters" not in kmodes_params:
            kmodes_params["n_clusters"] = 2

        self.bahc_max_iter = bahc_max_iter
        self.bahc_min_cluster_size = bahc_min_cluster_size
        self._bahc = BiasAwareHierarchicalClustering(
            KModes, bahc_max_iter, bahc_min_cluster_size, **kmodes_params
        )

    def fit(self, X, y):
        self._bahc.fit(X, y)
        self.n_clusters_ = self._bahc.n_clusters_
        self.labels_ = self._bahc.labels_
        self.scores_ = self._bahc.scores_
        self.cluster_tree_ = self._bahc.cluster_tree_
        return self

    def predict(self, X):
        return self._bahc.predict(X)
