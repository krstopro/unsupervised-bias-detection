from collections import deque
import heapq
from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from ._cluster_node import ClusterNode


class BiasAwareHierarchicalClustering(BaseEstimator, ClusterMixin):
    """
    Bias-Aware Hierarchical Clustering.

    BiasAwareHierarchicalClustering performs hierarchical clustering in a way that
    is aware of potential bias within the performance metric of interest.
    In each iteration, the method takes the cluster with the highest standard deviation
    of the performance metric among those clusters that were not taken in previous
    iterations. It then uses the `clustering_cls` to split the selected cluster
    into child clusters. The split is valid if the discrimination score of at least one
    child cluster is greater than or equal to the current discrimination score plus
    `margin` and all child clusters meet the minimum cluster size requirement.
    The discrimination score of a cluster is the difference between the mean of the
    performance metric on the complement of the cluster and the mean of the
    performance metric on the cluster. The method stops when the maximum number of
    iterations is reached or no more valid splits are possible.

    Parameters
    ----------
    clustering_cls : Type[ClusterMixin]
        The clustering class to use for each hierarchical split
        (e.g., sklearn.cluster.KMeans).
    bahc_max_iter : int
        Maximum number of iterations to run the hierarchical splitting procedure.
    bahc_min_cluster_size : int
        The minimum size a cluster must have to be further split.
    margin : float, optional (default=1e-5)
        Minimum score improvement required to split a cluster.
    **clustering_params : dict
        Additional hyperparameters to pass to clustering_cls upon instantiation.

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm.
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point. Lower labels correspond to
        higher discrimination scores.
    scores_ : ndarray of shape (n_clusters_,)
        Discrimination scores for each cluster.
    cluster_tree_ : ClusterNode
        The root node of the cluster tree.

    References
    ----------
    .. [1] J. Misztal-Radecka, B. Indurkhya, "Bias-Aware Hierarchical Clustering
           for detecting the discriminated groups of users in recommendation systems",
           Information Processing & Management, vol. 58, no. 3, May. 2021.

    Examples
    --------
    >>> from unsupervised_bias_detection.cluster import (
    ...     BiasAwareHierarchicalClustering,
    ... )
    >>> import numpy as np
    >>> from sklearn.cluster import KMeans
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> y = np.array([0, 0, 0, 10, 10, 10])
    >>> bahc = BiasAwareHierarchicalClustering(
    ...     clustering_cls=KMeans,
    ...     bahc_max_iter=1,
    ...     bahc_min_cluster_size=1,
    ...     n_clusters=2,
    ...     random_state=12,
    ... ).fit(X, y)
    >>> bahc.labels_
    array([0, 0, 0, 1, 1, 1], dtype=uint32)
    >>> bahc.scores_
    array([ 10., -10.])
    """

    _parameter_constraints: ClassVar[dict] = {
        "clustering_cls": [HasMethods(["fit"])],
        "bahc_max_iter": [Interval(Integral, 1, None, closed="left")],
        "bahc_min_cluster_size": [Interval(Integral, 1, None, closed="left")],
        "margin": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        clustering_cls: type[ClusterMixin],
        bahc_max_iter: int,
        bahc_min_cluster_size: int,
        margin: float = 1e-5,
        **clustering_params: Any,
    ):
        self.clustering_cls = clustering_cls
        self.bahc_max_iter = bahc_max_iter
        self.bahc_min_cluster_size = bahc_min_cluster_size
        self.margin = margin
        self.clustering_params = clustering_params

    def fit(self, X, y):
        """Compute bias-aware hierarchical clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : array-like of shape (n_samples)
            Performance metric values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()
        if not issubclass(self.clustering_cls, BaseEstimator):
            raise TypeError(
                f"clustering_cls must derive from BaseEstimator, "
                f"got {self.clustering_cls.__name__}"
            )
        if not issubclass(self.clustering_cls, ClusterMixin):
            raise TypeError(
                f"clustering_cls must derive from ClusterMixin, "
                f"got {self.clustering_cls.__name__}"
            )
        X, y = validate_data(
            self, X, y, reset=True, accept_large_sparse=False, order="C"
        )
        n_samples, _ = X.shape
        # We start with all samples being in a single cluster with label 0
        self.n_clusters_ = 1
        labels = np.zeros(n_samples, dtype=np.uint32)
        leaves = []
        label = 0
        # The entire dataset has a discrimination score of zero
        score = 0
        std = np.std(y)
        root = ClusterNode(label, score)
        self.cluster_tree_ = root
        heap = [(-std, label, root)]
        for _ in range(self.bahc_max_iter):
            if not heap:
                # If the heap is empty we stop iterating
                break
            # Take the cluster with the highest standard deviation of metric y
            _, label, node = heapq.heappop(heap)
            score = node.score
            cluster_indices = np.nonzero(labels == label)[0]
            X_cluster = X[cluster_indices]

            clustering_model = self.clustering_cls(**self.clustering_params)
            cluster_labels = clustering_model.fit_predict(X_cluster)

            if hasattr(clustering_model, "n_clusters_"):
                n_children = clustering_model.n_clusters_
            else:
                n_children = len(np.unique(cluster_labels))

            # We first check if all child clusters meet the minimum size requirement
            valid_split = True
            children_indices = []
            for i in range(n_children):
                child_indices = cluster_indices[np.nonzero(cluster_labels == i)[0]]
                if len(child_indices) >= self.bahc_min_cluster_size:
                    children_indices.append(child_indices)
                else:
                    valid_split = False
                    break

            # If all child clusters are of sufficient size,
            # we check if the score of any child cluster is
            # greater than or equal to the current score
            if valid_split:
                valid_split = False
                child_scores = []
                for child_indices in children_indices:
                    y_cluster = y[child_indices]
                    complement_mask = np.ones(n_samples, dtype=bool)
                    complement_mask[child_indices] = False
                    y_complement = y[complement_mask]
                    child_score = np.mean(y_complement) - np.mean(y_cluster)
                    if child_score >= score + self.margin:
                        valid_split = True
                    child_scores.append(child_score)

            if valid_split:
                # If the split is valid,
                # we create the child nodes and split the current node
                first_child_indices = children_indices[0]
                first_child_score = child_scores[0]
                first_child = ClusterNode(label, first_child_score)
                first_child_std = np.std(y[first_child_indices])
                # heapq implements a min-heap, so we negate std before pushing
                heapq.heappush(heap, (-first_child_std, label, first_child))
                labels[first_child_indices] = label
                children = [first_child]
                for i in range(1, n_children):
                    child_indices = children_indices[i]
                    child_score = child_scores[i]
                    child_node = ClusterNode(self.n_clusters_, child_score)
                    child_std = np.std(y[child_indices])
                    heapq.heappush(heap, (-child_std, self.n_clusters_, child_node))
                    labels[child_indices] = self.n_clusters_
                    children.append(child_node)
                    self.n_clusters_ += 1
                node.split(clustering_model, children)
            else:
                # Otherwise, we add the current node to the leaves
                leaves.append(node)

        leaves.extend([leaf for _, _, leaf in heap])
        leaf_scores = np.array([leaf.score for leaf in leaves])
        # We sort clusters by decreasing scores
        sorted_indices = np.argsort(-leaf_scores)
        self.scores_ = leaf_scores[sorted_indices]
        leaf_labels = np.array([leaf.label for leaf in leaves])
        leaf_labels = leaf_labels[sorted_indices]
        label_mapping = np.zeros(self.n_clusters_, dtype=np.uint32)
        label_mapping[leaf_labels] = np.arange(self.n_clusters_, dtype=np.uint32)
        self.labels_ = label_mapping[labels]
        for leaf in leaves:
            leaf.label = label_mapping[leaf.label]
        return self

    def predict(self, X):
        """Predict the cluster labels for the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        # Check if the clustering class has predict method
        if not hasattr(self.clustering_cls, "predict"):
            raise AttributeError(
                f"clustering_cls {self.clustering_cls.__name__} "
                f"does not have a predict method."
            )
        check_is_fitted(self, attributes="cluster_tree_")
        X = validate_data(self, X, reset=False, accept_large_sparse=False, order="C")
        n_samples, _ = X.shape
        labels = np.zeros(n_samples, dtype=np.uint32)
        queue = deque([(self.cluster_tree_, np.arange(n_samples))])
        while queue:
            node, indices = queue.popleft()
            if node.is_leaf:
                labels[indices] = node.label
            else:
                cluster = X[indices]
                clustering_model = node.clustering_model
                cluster_labels = clustering_model.predict(cluster)
                if hasattr(clustering_model, "n_clusters_"):
                    n_clusters = clustering_model.n_clusters_
                else:
                    n_clusters = len(np.unique(cluster_labels))
                for i in range(n_clusters):
                    child_indices = indices[np.nonzero(cluster_labels == i)[0]]
                    if child_indices.size > 0:
                        queue.append((node.children[i], child_indices))
        return labels
