"""Tests for numerical bias-aware hierarchical clustering algorithms."""

import numpy as np
import pytest

from unsupervised_bias_detection.cluster import (
    BiasAwareHierarchicalClustering,
    BiasAwareHierarchicalKMeans,
)

NUMERICAL_ALGORITHMS = [BiasAwareHierarchicalKMeans]
NUMERICAL_ALGORITHMS_WITH_PREDICT = [BiasAwareHierarchicalKMeans]


@pytest.fixture
def data():
    """Generate random data with numerical features for testing."""
    rng = np.random.RandomState(12)
    X = rng.rand(20, 10)
    y = rng.rand(20)
    return X, y


@pytest.mark.parametrize("algorithm", NUMERICAL_ALGORITHMS, ids=lambda a: a.__name__)
def test_shapes(algorithm: type[BiasAwareHierarchicalClustering], data):
    """Test that labels and scores have the right shapes."""
    X, y = data
    bahc = algorithm(bahc_max_iter=5, bahc_min_cluster_size=2)
    bahc.fit(X, y)
    assert len(bahc.labels_) == len(
        X
    ), f"labels_ length {len(bahc.labels_)} does not match X length {len(X)}"
    assert len(bahc.scores_) == bahc.n_clusters_, (
        f"scores_ length {len(bahc.scores_)} "
        f"does not match n_clusters_ {bahc.n_clusters_}"
    )


@pytest.mark.parametrize("algorithm", NUMERICAL_ALGORITHMS, ids=lambda a: a.__name__)
def test_labels(algorithm: type[BiasAwareHierarchicalClustering], data):
    """Test that unique labels are np.arange(n_clusters_)."""
    X, y = data
    bahc = algorithm(bahc_max_iter=4, bahc_min_cluster_size=2)
    bahc.fit(X, y)
    unique_labels = np.unique(bahc.labels_)
    assert np.array_equal(unique_labels, np.arange(bahc.n_clusters_)), (
        f"Unique labels {unique_labels} do not match "
        f"expected range {np.arange(bahc.n_clusters_)} "
        f"for n_clusters_={bahc.n_clusters_}"
    )


@pytest.mark.parametrize("algorithm", NUMERICAL_ALGORITHMS, ids=lambda a: a.__name__)
def test_cluster_sizes(algorithm: type[BiasAwareHierarchicalClustering], data):
    """Test that all clusters have at least bahc_min_cluster_size samples."""
    X, y = data
    min_cluster_size = 5
    model = algorithm(bahc_max_iter=5, bahc_min_cluster_size=min_cluster_size)
    model.fit(X, y)
    sizes = np.bincount(model.labels_)
    assert np.all(
        sizes >= min_cluster_size
    ), f"Cluster sizes found: {sizes}, expected each to be >= {min_cluster_size}"


@pytest.mark.parametrize("algorithm", NUMERICAL_ALGORITHMS, ids=lambda a: a.__name__)
def test_constant_metric(algorithm: type[BiasAwareHierarchicalClustering]):
    """Test that there is only one cluster with a score of 0 for constant metric."""
    rng = np.random.RandomState(12)
    X = rng.rand(20, 10)
    y = np.full(20, rng.rand())
    model = algorithm(bahc_max_iter=5, bahc_min_cluster_size=2)
    model.fit(X, y)
    assert model.n_clusters_ == 1, f"Expected 1 cluster, found {model.n_clusters_}"
    assert model.scores_[0] == 0, f"Expected score of 0, found {model.scores_[0]}"


@pytest.mark.parametrize("algorithm", NUMERICAL_ALGORITHMS, ids=lambda a: a.__name__)
def test_scores(algorithm: type[BiasAwareHierarchicalClustering], data):
    """Test that scores are computed correctly."""
    X, y = data
    model = algorithm(bahc_max_iter=5, bahc_min_cluster_size=2)
    model.fit(X, y)
    # TODO: Check this!!!
    for i in range(model.n_clusters_):
        cluster_indices = np.arange(20)[model.labels_ == i]
        complement_indices = np.arange(20)[model.labels_ != i]
        score = np.mean(y[complement_indices]) - np.mean(y[cluster_indices])
        assert model.scores_[i] == score


@pytest.mark.parametrize("algorithm", NUMERICAL_ALGORITHMS, ids=lambda a: a.__name__)
def test_scores_are_sorted(algorithm: type[BiasAwareHierarchicalClustering], data):
    """Test that scores are sorted in descending order."""
    X, y = data
    model = algorithm(bahc_max_iter=5, bahc_min_cluster_size=2)
    model.fit(X, y)
    assert np.all(
        model.scores_[:-1] >= model.scores_[1:]
    ), "Scores are not sorted in descending order"


@pytest.mark.parametrize(
    "algorithm", NUMERICAL_ALGORITHMS_WITH_PREDICT, ids=lambda a: a.__name__
)
def test_predict(algorithm: type[BiasAwareHierarchicalClustering], data):
    """Test that predict returns the same labels on the data used to fit the model."""
    X, y = data
    model = algorithm(bahc_max_iter=5, bahc_min_cluster_size=2)
    model.fit(X, y)
    assert np.array_equal(
        model.predict(X), model.labels_
    ), "Predict does not return the same labels on the data used to fit the model"
