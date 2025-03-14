import heapq
from numbers import Integral
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import validate_data
from typing import Any, Type


class BiasAwareHierarchicalClustering(BaseEstimator, ClusterMixin):
    """TODO: Add docstring.

    References
    ----------
    .. [1] J. Misztal-Radecka, B. Indurkhya, "Bias-Aware Hierarchical Clustering for detecting the discriminated
           groups of users in recommendation systems", Information Processing & Management, vol. 58, no. 3, May. 2021.
    """

    _parameter_constraints: dict = {
        "bahc_max_iter": [Interval(Integral, 1, None, closed="left")],
        "bahc_min_cluster_size": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        clustering_cls: Type[ClusterMixin],
        bahc_max_iter: int,
        bahc_min_cluster_size: int,
        **clustering_params: Any,
    ):
        self.clustering_cls = clustering_cls
        self.bahc_max_iter = bahc_max_iter
        self.bahc_min_cluster_size = bahc_min_cluster_size
        self.clustering_params = clustering_params

    def fit(self, X, y):
        """Compute bias-aware hierarchical clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : array-like of shape (n_samples)
            Metric values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = validate_data(
            self,
            X,
            y,
            reset=False,
            accept_large_sparse=False,
            order="C",
        )
        n_samples, _ = X.shape
        # We start with all samples in a single cluster
        self.n_clusters_ = 1
        # We assign all samples a label of zero
        labels = np.zeros(n_samples, dtype=np.uint32)
        clusters = []
        scores = []
        label = 0
        # The entire dataset has a discrimination score of zero
        score = 0
        heap = [(None, label, score)]
        for _ in range(self.bahc_max_iter):
            if not heap:
                # If the heap is empty we stop iterating
                break
            # Take the cluster with the highest standard deviation of metric y
            _, label, score = heapq.heappop(heap)
            cluster_indices = np.nonzero(labels == label)[0]
            cluster = X[cluster_indices]

            clustering_model = self.clustering_cls(**self.clustering_params)
            cluster_labels = clustering_model.fit_predict(cluster)

            # TODO: Generalize for more than 2 clusters
            # Can do this by checking clustering_model.n_clusters_ (if it exists)
            # or by checking the number of unique values in cluster_labels
            indices0 = cluster_indices[np.nonzero(cluster_labels == 0)[0]]
            indices1 = cluster_indices[np.nonzero(cluster_labels == 1)[0]]
            if (
                len(indices0) >= self.bahc_min_cluster_size
                and len(indices1) >= self.bahc_min_cluster_size
            ):
                # We calculate the discrimination scores using formula (1) in [1]
                mask0 = np.ones(n_samples, dtype=bool)
                mask0[indices0] = False
                score0 = np.mean(y[mask0]) - np.mean(y[indices0])
                mask1 = np.ones(n_samples, dtype=bool)
                mask1[indices1] = False
                score1 = np.mean(y[mask1]) - np.mean(y[indices1])
                if max(score0, score1) >= score:
                    # heapq implements min-heap
                    # so we have to negate std before pushing
                    std0 = np.std(y[indices0])
                    heapq.heappush(heap, (-std0, label, score0))
                    std1 = np.std(y[indices1])
                    heapq.heappush(heap, (-std1, self.n_clusters_, score1))
                    labels[indices1] = self.n_clusters_
                    self.n_clusters_ += 1
                else:
                    clusters.append(label)
                    scores.append(score)
            else:
                clusters.append(label)
                scores.append(score)
        if heap:
            clusters = np.concatenate([clusters, [label for _, label, _ in heap]])
            scores = np.concatenate([scores, [score for _, _, score in heap]])
        else:
            clusters = np.array(clusters)
            scores = np.array(scores)

        # We sort clusters by decreasing scores
        indices = np.argsort(-scores)
        clusters = clusters[indices]
        self.scores_ = scores[indices]
        mapping = np.zeros(self.n_clusters_, dtype=np.uint32)
        mapping[clusters] = np.arange(self.n_clusters_, dtype=np.uint32)
        self.labels_ = mapping[labels]
        return self
